#!/usr/bin/env python3
import os
import time
import json
import http
import requests

from setup import setup_logging

TOTAL_TRIES = 3
RETRY_DELAY = 10
RETRY_CODES = [
    http.HTTPStatus.REQUEST_TIMEOUT,
    http.HTTPStatus.TOO_MANY_REQUESTS,
    http.HTTPStatus.INTERNAL_SERVER_ERROR,
    http.HTTPStatus.BAD_GATEWAY,
    http.HTTPStatus.SERVICE_UNAVAILABLE,
    http.HTTPStatus.GATEWAY_TIMEOUT
]

def with_retries(func):
    def wrapper(*args, **kwargs):
        tries = 0
        while tries < TOTAL_TRIES:
            try:
                return func(*args, **kwargs)
            except requests.HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
                if http_err.response.status_code in RETRY_CODES:
                    tries += 1
                    logger.info(f"Retrying in {RETRY_DELAY} seconds")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    logger.error(f"Failed to send message: {http_err}")
                    break
            except requests.RequestException as req_err:
                logger.error(f"Request error occurred: {req_err}")
                break
        logger.error("Failed to send message. Maximum retries reached")
    return wrapper


logger = setup_logging('logs/telelogger', 'TELELOGGER')

class TeleLogger:
    def __init__(self, token: str, chat_id: str, use_requests_logs: bool = False):
        if not token:
            raise ValueError("Token is required")
        if not chat_id:
            raise ValueError("Chat ID is required")
        self._token = token
        self._chat_id = chat_id
        self._url = f"https://api.telegram.org/bot{self._token}/"
        self._send_message_url = f"{self._url}sendMessage"
        self._send_media_url = f"{self._url}sendMediaGroup"
        if not use_requests_logs:
            import logging
            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    @with_retries
    def send_message(self, message: str):
        send_data = {
            'chat_id': self._chat_id,
            'text': message
        }
        response = requests.post(self._send_message_url, data=send_data)
        response.raise_for_status()

    @with_retries
    def send_message_with_media(self, media: list, message: str | None = None):
        send_data = {
            'chat_id': self._chat_id,
            'media': json.dumps(
                [
                    {
                        'type': 'photo',
                        'media': f"attach://image_{i}.png",
                        'caption': message if i == 0 else ''
                    } for i in range(len(media))
                ]
            )
        }
        files = {f"image_{i}.png": open(media[i], 'rb') for i in range(len(media))}
        response = requests.post(self._send_media_url, data=send_data, files=files)
        response.raise_for_status()

    @staticmethod
    def send_in_thread(func, *args, **kwargs):
        import threading
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

if __name__ == '__main__':
    config = {
        "token": "",
        "chat_id": "",
        "use_requests_logs": False
    }
    logger.info("Starting TeleLogger")
    telelogger = TeleLogger(**config)

    import numpy as np
    import matplotlib.pyplot as plt
    random_data = np.random.rand(250, 250)
    # save the image
    plt.imsave("random_data.png", random_data)
    # telelogger.send_message_with_media(["random_data.png"], "Random data image")
    TeleLogger.send_in_thread(telelogger.send_message_with_media, media=["random_data.png", "random_data.png"], message="Random data image")
