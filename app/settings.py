import os
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_CONFIG = {
    'key': os.getenv('API_TOKEN'),
    'url': os.getenv('WEBHOOK_URL')
}

RESPONSE_CODES = {}

AWS_CONFIG = {
    'key': os.getenv('AWS_KEY'),
    'secret': os.getenv('AWS_SECRET'),
    'bucket': os.getenv('AWS_BUCKET'),
    'region': os.getenv('AWS_REGION'),
    'sqs': os.getenv('AWS_SQS_URL'),
    'sqs_handler': 'handler.json'
}

SERVER = {
    'main_id': os.getenv('MAIN_ID')
}

SHUTDOWN_TIMINGS = {
    'minutes': os.getenv('MINUTES'),
    'intermediate': os.getenv('INTERMEDIATE')
}

BALANCER = {
    'main_for': os.getenv('MAIN_FOR')
}

BALANCER_SERVER_TYPES = ['cloning', 'tts', 'lipsync', 'tortoise']

BALANCER_TIMES = {
    'cloning': 420,
    'tts': 3,
    'lipsync': 20,
    'tortoise': 10
}

BALANCER_RESOURCES = {
    'cloning': {
        'cpu': 100,
        'ram': 3550,
        'gpu': 3658
    },
    'tts': {
        'cpu': 17,
        'ram': 3335,
        'gpu': 2060
    },
    'lipsync': {
        'cpu': 100,
        'ram': 22000,
        'gpu': 14000
    },
    'tortoise': {
        'cpu': 77,
        'ram': 9582,
        'gpu': 15200
    }
}

SERVER_RESOURCES_TOTAL = {
    'gpu': os.getenv('GPU_TOTAL'),
    'ram': os.getenv('RAM_TOTAL')
}

VIDEO_CONFIG = {
    'seq_chunk': os.getenv('SEQ_CHUNK')
}