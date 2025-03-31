import smtplib
# SMTP Settings
smtp_server = "smtp.dreamhost.com"
smtp_port = 587
smtp_username = "notifications@zeibari.net"
smtp_password = "xcj7yak0vev!ZME3gxh"
from_email = "notifications@zeibari.net"
to_email = "kevork@zeibari.com"  # Email from agents.json
# Connect to the server
try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    print("Authentication successful!")
    
    # Try sending a test email - using string formatting compatible with Python 2.7
    message = "From: {0}\nTo: {1}\nSubject: Test Email\n\nThis is a test email.".format(from_email, to_email)
    server.sendmail(from_email, to_email, message)
    print("Test email sent successfully!")
    
    server.quit()
except Exception as e:
    print("Error: {}".format(e))