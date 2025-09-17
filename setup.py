from setuptools import setup, find_packages

setup(
    name="realtime-market-intelligence",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "sqlalchemy==2.0.23",
        "psycopg2-binary==2.9.9",
        "kafka-python==2.0.2",
        "python-dotenv==1.0.0",
    ],
)