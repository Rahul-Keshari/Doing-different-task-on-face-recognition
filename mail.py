import smtplib
import os
from smtplib import SMTPException


def send_mail():
    
    passwd=os.environ['PASSWD']
    # set up the SMTP server
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    #s.starttls()
    s.login('your_email' , passwd)
    try:
        sender = 'your_email'
        receivers = ['sender_email']
        message = """From: Face Detection Program!! 
To: Owner
Subject: FaceDetected!!
Warm Welcome from Tech-Trollers!! Rahul's face was detected!!
"""
        
        for i  in receivers:
            s.sendmail(sender, i , message)         
            print("Successfully sent email to -->"+ i)
    except Exception as E:
        print("Error: unable to send email\n")
