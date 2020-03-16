#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys
import time


def main():
 # 第三方 SMTP 服务

    mail_host="smtp.qq.com"  #设置服务器
    mail_user="1347481725@qq.com"    #用户名
    mail_pass="awmwznsyfpwliecg"   #口令 
 
 
    sender = '1347481725@qq.com'
    receivers = ['liuyu@hit-mtlab.net']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
 
    print(sys.argv[1])
    
    message = MIMEText(sys.argv[1], 'plain', 'utf-8')
    message['From'] = Header("Best_BLEU", 'utf-8')
    message['To'] =  Header("BLEU", 'utf-8')
 
    subject = 'Best_BLEU'
    message['Subject'] = Header(subject, 'utf-8')
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user,mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException:
        print("Error: 无法发送邮件")


if __name__ == '__main__':
    main()