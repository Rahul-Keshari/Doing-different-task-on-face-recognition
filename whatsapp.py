from twilio.rest import Client

def whatsapp():
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN'] 
    client = Client(account_sid, auth_token) 
    ss=['whatsapp:+918276972706']
    for i in ss:
        message = client.messages.create( 
                              from_='whatsapp:+14155238886',  
                              body="Type your message here!",      
                              to= i 
                          ) 
        print(message.sid)
