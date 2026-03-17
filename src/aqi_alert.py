"""
AQI Alert System
=================
Checks predicted AQI for configured cities and sends an email
alert when any city crosses a defined threshold.

Setup (one-time):
  Copy alerts_config.example.json -> alerts_config.json and fill in your details.

Schedule daily via Windows Task Scheduler:
  Action: python D:\Projects\Predicting Air Quality Index\src\aqi_alert.py
  Trigger: Daily at 07:00

Gmail setup:
  1. Enable 2-Factor Authentication on your Google account
  2. Go to: Google Account > Security > App Passwords
  3. Generate a new App Password and paste it as smtp_password

Usage:
    python src/aqi_alert.py                   # use alerts_config.json
    python src/aqi_alert.py --dry-run         # print alerts without sending email
    python src/aqi_alert.py --test-email      # send a test email
"""

import os
import sys
import json
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from realtime_aqi_pipeline import predict_aqi, aqi_to_category, aqi_to_color, AQI_CATEGORIES, INDIA_CITIES

_ROOT       = os.path.join(os.path.dirname(__file__), '..')
CONFIG_PATH = os.path.join(_ROOT, 'alerts_config.json')
LOG_PATH    = os.path.join(_ROOT, 'outputs', 'alert_log.csv')

DEFAULT_CONFIG = {
    "smtp_host"    : "smtp.gmail.com",
    "smtp_port"    : 587,
    "smtp_user"    : "your_email@gmail.com",
    "smtp_password": "your_app_password",
    "to_email"     : "your_email@gmail.com",
    "threshold_aqi": 150,
    "cities": [
        {"name": "Delhi",     "lat": 28.6139, "lng": 77.2090},
        {"name": "Mumbai",    "lat": 19.0760, "lng": 72.8777},
        {"name": "Kolkata",   "lat": 22.5726, "lng": 88.3639},
        {"name": "Chennai",   "lat": 13.0827, "lng": 80.2707},
        {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946},
    ]
}

# Health advice per AQI category
HEALTH_ADVICE = {
    'Good'        : "Air quality is satisfactory. Enjoy outdoor activities.",
    'Satisfactory': "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    'Moderate'    : "Members of sensitive groups may experience health effects. General public less likely to be affected.",
    'Poor'        : "Everyone may begin to experience health effects. Sensitive groups should limit outdoor activity.",
    'Very Poor'   : "Health alert — everyone may experience serious health effects. Avoid outdoor exertion.",
    'Severe'      : "Health warning of emergency conditions. Everyone should avoid outdoor activity. Wear N95 mask if outdoors.",
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        # Write example config
        example_path = os.path.join(_ROOT, 'alerts_config.example.json')
        with open(example_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Config not found. Example written to: alerts_config.example.json")
        print("Copy it to alerts_config.json and fill in your credentials.")
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


def check_cities(config):
    """Run predictions for all configured cities. Returns list of alerts."""
    threshold = config.get('threshold_aqi', 150)
    alerts = []
    all_results = []

    for city in config.get('cities', []):
        name = city['name']
        lat, lng = city['lat'], city['lng']
        aqi, cat = predict_aqi(lat, lng)
        all_results.append({'city': name, 'aqi': aqi, 'category': cat})
        if aqi >= threshold:
            alerts.append({'city': name, 'lat': lat, 'lng': lng,
                           'aqi': aqi, 'category': cat,
                           'advice': HEALTH_ADVICE.get(cat, '')})

    return alerts, all_results


def build_email_html(alerts, all_results, threshold):
    cat_colors = {c: col for _, _, c, col in AQI_CATEGORIES}
    rows = ""
    for r in all_results:
        color = cat_colors.get(r['category'], '#999')
        flag = " ** ALERT **" if r['aqi'] >= threshold else ""
        rows += (f"<tr><td style='padding:6px 12px'>{r['city']}</td>"
                 f"<td style='padding:6px 12px;background:{color};color:white;"
                 f"text-align:center;border-radius:4px'>"
                 f"<b>{r['aqi']:.0f}</b></td>"
                 f"<td style='padding:6px 12px'>{r['category']}{flag}</td>"
                 f"<td style='padding:6px 12px;font-size:0.85em'>"
                 f"{HEALTH_ADVICE.get(r['category'],'')}</td></tr>")

    alert_cities = ', '.join(a['city'] for a in alerts)
    html = f"""
<html><body style='font-family:Arial,sans-serif;color:#333'>
<h2 style='color:#e74c3c'>AQI Alert — {len(alerts)} city/cities exceeded threshold {threshold}</h2>
<p><b>Triggered at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><b>Cities above threshold:</b> {alert_cities}</p>
<table border='0' cellpadding='0' cellspacing='4'
       style='border-collapse:separate;border-spacing:4px'>
  <tr style='background:#eee'>
    <th style='padding:6px 12px;text-align:left'>City</th>
    <th style='padding:6px 12px'>AQI</th>
    <th style='padding:6px 12px;text-align:left'>Category</th>
    <th style='padding:6px 12px;text-align:left'>Advice</th>
  </tr>
  {rows}
</table>
<br>
<p style='font-size:0.8em;color:#888'>
  India AQI Prediction System &mdash; alerts_config.json threshold = {threshold}
</p>
</body></html>"""
    return html


def send_email(config, subject, html_body, dry_run=False):
    if dry_run:
        print(f"\n[DRY RUN] Would send email:")
        print(f"  To      : {config['to_email']}")
        print(f"  Subject : {subject}")
        return True

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = config['smtp_user']
    msg['To']      = config['to_email']
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['smtp_user'], config['smtp_password'])
            server.sendmail(config['smtp_user'], config['to_email'], msg.as_string())
        print(f"  Email sent to {config['to_email']}")
        return True
    except smtplib.SMTPException as e:
        print(f"  Email failed: {e}")
        return False


def log_results(all_results, alerts_sent):
    import csv
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'city', 'aqi', 'category', 'alert_sent'])
        if not file_exists:
            writer.writeheader()
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        for r in all_results:
            writer.writerow({
                'timestamp'  : ts,
                'city'       : r['city'],
                'aqi'        : r['aqi'],
                'category'   : r['category'],
                'alert_sent' : alerts_sent,
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AQI Alert System')
    parser.add_argument('--dry-run',    action='store_true', help='Print alert without sending email')
    parser.add_argument('--test-email', action='store_true', help='Send a test email')
    args = parser.parse_args()

    config = load_config()
    if config is None:
        sys.exit(1)

    if args.test_email:
        html = build_email_html([], [{'city':'Test','aqi':50,'category':'Good'}],
                                config.get('threshold_aqi', 150))
        send_email(config, "AQI Alert — Test Email", html, dry_run=args.dry_run)
        sys.exit(0)

    print(f"Checking AQI for {len(config.get('cities',[]))} cities "
          f"(threshold = {config.get('threshold_aqi', 150)}) ...")
    alerts, all_results = check_cities(config)

    print(f"\n  {'City':<15} {'AQI':>6}  Category")
    print("  " + "-" * 35)
    for r in all_results:
        flag = " ** ALERT **" if r['aqi'] >= config.get('threshold_aqi', 150) else ""
        print(f"  {r['city']:<15} {r['aqi']:>6.0f}  {r['category']}{flag}")

    alerts_sent = False
    if alerts:
        print(f"\n  {len(alerts)} city/cities exceeded threshold!")
        subject = (f"AQI Alert — {', '.join(a['city'] for a in alerts)} "
                   f"exceeded AQI {config.get('threshold_aqi',150)}")
        html = build_email_html(alerts, all_results, config.get('threshold_aqi', 150))
        alerts_sent = send_email(config, subject, html, dry_run=args.dry_run)
    else:
        print(f"\n  All cities below threshold {config.get('threshold_aqi',150)}. No alert sent.")

    log_results(all_results, alerts_sent)
    print(f"  Results logged -> outputs/alert_log.csv")
