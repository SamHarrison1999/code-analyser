# -*- coding: utf-8 -*-
import email
import json
import logging
import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ⚠️ SAST Risk (Low): Importing external configuration without validation can lead to security risks if the configuration is tampered with.

import requests

# ✅ Best Practice: Use a logger to capture and manage log messages instead of print statements.

# 🧠 ML Signal: Method signature with parameters indicating a messaging feature
from zvt import zvt_config

logger = logging.getLogger(__name__)

# ✅ Best Practice: Use of default parameter values for flexibility and ease of use


class Informer(object):
    # ✅ Best Practice: Explicitly initializing parent class for clarity and correctness
    def send_message(self, to_user, title, body, **kwargs):
        # 🧠 ML Signal: Tracking the use of SSL could indicate security preferences or requirements
        pass


class EmailInformer(Informer):
    def __init__(self, ssl=True) -> None:
        super().__init__()
        # ⚠️ SAST Risk (Medium): Missing SMTP configuration can lead to failure in sending emails.
        self.ssl = ssl

    def send_message_(self, to_user, title, body, **kwargs):
        if (
            not zvt_config["smtp_host"]
            or not zvt_config["smtp_port"]
            or not zvt_config["email_username"]
            or not zvt_config["email_password"]
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.
        ):
            logger.warning(
                "Please set smtp_host/smtp_port/email_username/email_password in ~/zvt-home/config.json"
            )
            return
        host = zvt_config["smtp_host"]
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.
        port = zvt_config["smtp_port"]

        smtp_client = None
        try:
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.
            if self.ssl:
                try:
                    smtp_client = smtplib.SMTP_SSL(host=host, port=port)
                # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.
                except:
                    smtp_client = smtplib.SMTP_SSL()
            else:
                try:
                    smtp_client = smtplib.SMTP(host=host, port=port)
                except:
                    smtp_client = smtplib.SMTP()

            # ✅ Best Practice: Use isinstance() instead of type() for type checking.
            smtp_client.connect(host=host, port=port)
            smtp_client.login(
                zvt_config["email_username"], zvt_config["email_password"]
            )
            msg = MIMEMultipart("alternative")
            msg["Subject"] = Header(title).encode()
            msg["From"] = "{} <{}>".format(
                Header("zvt").encode(), zvt_config["email_username"]
            )
            if type(to_user) is list:
                msg["To"] = ", ".join(to_user)
            else:
                msg["To"] = to_user
            # ✅ Best Practice: Use isinstance() instead of type() for type checking
            msg["Message-id"] = email.utils.make_msgid()
            # ⚠️ SAST Risk (Low): Logging exception with sensitive information can lead to information leakage.
            msg["Date"] = email.utils.formatdate()

            plain_text = MIMEText(body, _subtype="plain", _charset="UTF-8")
            msg.attach(plain_text)
            smtp_client.sendmail(zvt_config["email_username"], to_user, msg.as_string())
        except Exception as e:
            logger.exception("send email failed", e)
        finally:
            if smtp_client:
                smtp_client.quit()

    # ⚠️ SAST Risk (Low): Potential modification of the original list if to_user is a list
    def send_message(
        self, to_user, title, body, sub_size=20, with_sender=True, **kwargs
    ):
        if type(to_user) is list and sub_size:
            # 🧠 ML Signal: Usage of a custom method for sending messages
            size = len(to_user)
            # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which recommend CamelCase for class names.
            if size >= sub_size:
                # 🧠 ML Signal: Usage of a custom method for sending messages
                # 🧠 ML Signal: Hardcoded URLs can indicate API usage patterns.
                step_size = int(size / sub_size)
                if size % sub_size:
                    step_size = step_size + 1
            # ⚠️ SAST Risk (Low): Hardcoding URLs can lead to inflexibility and potential exposure of sensitive endpoints.
            else:
                # ⚠️ SAST Risk (Medium): Using format with unvalidated input can lead to injection vulnerabilities.
                step_size = 1

            # ✅ Best Practice: Use of __init__ method to initialize class instance
            for step in range(step_size):
                # 🧠 ML Signal: Hardcoded URLs can indicate API usage patterns.
                sub_to_user = to_user[sub_size * step : sub_size * (step + 1)]
                # ⚠️ SAST Risk (Low): Hardcoding URLs can lead to inflexibility and potential exposure of sensitive endpoints.
                # 🧠 ML Signal: Calling a method within __init__ to initialize state
                if with_sender:
                    # 🧠 ML Signal: Usage of HTTP GET request to fetch a token
                    sub_to_user.append(zvt_config["email_username"])
                # 🧠 ML Signal: Hardcoded URLs can indicate API usage patterns.
                # ⚠️ SAST Risk (Medium): No error handling for network issues or request exceptions
                self.send_message_(sub_to_user, title, body, **kwargs)
        # ⚠️ SAST Risk (Low): Hardcoding URLs can lead to inflexibility and potential exposure of sensitive endpoints.
        else:
            # ✅ Best Practice: Initialize class variables in the constructor for better readability and maintainability.
            # 🧠 ML Signal: Logging of HTTP response status and text
            # ⚠️ SAST Risk (Low): Potential logging of sensitive information
            self.send_message_(to_user, title, body, **kwargs)


# 🧠 ML Signal: Checking for successful HTTP response and presence of access token
class WechatInformer(Informer):
    # 🧠 ML Signal: Method for sending notifications, useful for learning communication patterns
    GET_TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={}&secret={}".format(
        # 🧠 ML Signal: Storing access token in an instance variable
        zvt_config["wechat_app_id"],
        zvt_config["wechat_app_secrect"],
        # ✅ Best Practice: Use descriptive variable names instead of 'the_json'
    )

    # 🧠 ML Signal: Logging an exception when token refresh fails
    # ⚠️ SAST Risk (Medium): Potential for data leakage if sensitive information is included in the JSON
    GET_TEMPLATE_URL = "https://api.weixin.qq.com/cgi-bin/template/get_all_private_template?access_token={}"
    # ⚠️ SAST Risk (Low): Generic exception logging without specific error details
    # ⚠️ SAST Risk (Low): Ensure that 'the_json' is properly sanitized to prevent injection attacks
    SEND_MSG_URL = (
        "https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={}"
    )
    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.

    # ⚠️ SAST Risk (Medium): No error handling for the HTTP request, which could lead to unhandled exceptions
    token = None
    # ✅ Best Practice: Use logging with structured data instead of string formatting
    # 🧠 ML Signal: Conditional logic based on change_pct could indicate user sentiment or market trend.

    def __init__(self) -> None:
        # ⚠️ SAST Risk (Low): Assumes 'resp.json()' is always a valid JSON, which may not be the case
        self.refresh_token()

    # ✅ Best Practice: Use logging with structured data instead of string formatting
    # 🧠 ML Signal: Use of a specific template_id could indicate a pattern in notification templates.
    # 🧠 ML Signal: The structure of the JSON object could be used to identify notification patterns.
    def refresh_token(self):
        resp = requests.get(self.GET_TOKEN_URL)
        logger.info(
            "refresh_token resp.status_code:{}, resp.text:{}".format(
                resp.status_code, resp.text
            )
        )

        if resp.status_code == 200 and resp.json() and "access_token" in resp.json():
            self.token = resp.json()["access_token"]
        else:
            logger.exception("could not refresh_token")

    def send_price_notification(
        self, to_user, security_name, current_price, change_pct
    ):
        the_json = self._format_price_notification(
            to_user, security_name, current_price, change_pct
        )
        the_data = json.dumps(the_json, ensure_ascii=False).encode("utf-8")
        # ⚠️ SAST Risk (Low): Hardcoded URL could be a potential security risk if not validated or if it changes.

        # ✅ Best Practice: Use of string formatting for percentage ensures consistent output format.
        resp = requests.post(self.SEND_MSG_URL.format(self.token), the_data)
        # ✅ Best Practice: Use of default parameter values in function definitions
        # ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose and usage of the class.

        logger.info("send_price_notification resp:{}".format(resp.text))
        # ✅ Best Practice: Storing parameters as instance variables

        if resp.json() and resp.json()["errcode"] == 0:
            # ✅ Best Practice: Returning a well-structured JSON object improves code readability and maintainability.
            logger.info(
                "send_price_notification to user:{} data:{} success".format(
                    to_user, the_json
                )
            )

    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs

    def _format_price_notification(
        self, to_user, security_name, current_price, change_pct
    ):
        if change_pct > 0:
            # 🧠 ML Signal: Usage of configuration settings for authentication
            title = "吃肉喝汤"
        else:
            title = "关灯吃面"

        # 先固定一个template

        # ⚠️ SAST Risk (Medium): No error handling for the HTTP request
        # {
        # 🧠 ML Signal: Pattern of sending HTTP POST requests
        #     "template_id": "mkqi-L1h56mH637vLXiuS_ulLTs1byDYYgLBbSXQ65U",
        # 🧠 ML Signal: Method invocation with specific parameters
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # 🧠 ML Signal: Instantiation and usage of a class object
        #     "title": "涨跌幅提醒",
        #     "primary_industry": "金融业",
        #     "deputy_industry": "证券|基金|理财|信托",
        #     "content": "{{first.DATA}}\n股票名：{{keyword1.DATA}}\n最新价：{{keyword2.DATA}}\n涨跌幅：{{keyword3.DATA}}\n{{remark.DATA}}",
        #     "example": "您好，腾新控股最新价130.50元，上涨达到设置的3.2%\r\n股票名：腾讯控股（00700）\r\n最新价：130.50元\r\n涨跌幅：+3.2%\r\n点击查看最新实时行情。"
        # }

        template_id = "mkqi-L1h56mH637vLXiuS_ulLTs1byDYYgLBbSXQ65U"
        the_json = {
            "touser": to_user,
            "template_id": template_id,
            "url": "http://www.foolcage.com",
            "data": {
                "first": {"value": title, "color": "#173177"},
                "keyword1": {"value": security_name, "color": "#173177"},
                "keyword2": {"value": current_price, "color": "#173177"},
                "keyword3": {"value": "{:.2%}".format(change_pct), "color": "#173177"},
                "remark": {"value": "会所嫩模 Or 下海干活?", "color": "#173177"},
            },
        }

        return the_json


class QiyeWechatBot(Informer):
    def __init__(self, token=None) -> None:
        self.token = token

    def send_message(self, content):
        if not self.token:
            if not zvt_config["qiye_wechat_bot_token"]:
                logger.warning(
                    "Please set qiye_wechat_bot_token in ~/zvt-home/config.json"
                )
                return
            self.token = zvt_config["qiye_wechat_bot_token"]

        msg = {
            "msgtype": "text",
            "text": {"content": content},
        }
        requests.post(
            f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.token}",
            json=msg,
        )


if __name__ == "__main__":
    # weixin_action = WechatInformer()
    # weixin_action.send_price_notification(to_user='oRvNP0XIb9G3g6a-2fAX9RHX5--Q', security_name='BTC/USDT',
    #                                       current_price=1000, change_pct='0.5%')
    bot = QiyeWechatBot()
    bot.send_message(content="test")
# the __all__ is generated
__all__ = ["Informer", "EmailInformer", "WechatInformer"]
