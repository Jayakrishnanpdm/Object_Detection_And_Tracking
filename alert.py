import smtplib
import cv2
import telegram
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ========== EMAIL ALERT ==========
def send_email_alert(image_path, message):
    sender_email = "jay.work.developer@gmail.com"
    sender_password = "anje vvtv dyiw oglg"  # Use App Password if Gmail
    receiver_email = "jayakrishnan9446718962@gmail.com"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "⚠ Theft Alert!"

    msg.attach(MIMEText(message, "plain"))

    with open(image_path, "rb") as img:
        msg.attach(MIMEImage(img.read(), name="alert.jpg"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("✅ Email alert sent!")
    except Exception as e:
        print("❌ Email failed:", e)


# ========== TELEGRAM ALERT ==========
def send_telegram_alert(image_path, message):
    bot_token = "8442626184:AAFpFGCJx7iHT1pFBKEKhK5aUAusEHn-zks"
    chat_id = "1348434893"

    bot = telegram.Bot(token=bot_token)
    with open(image_path, "rb") as img:
        bot.send_photo(chat_id=chat_id, photo=img, caption=message)
    print("✅ Telegram alert sent!")
