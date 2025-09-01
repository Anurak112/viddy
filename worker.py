"""
RQ worker for ssstik-clone. When run, this script connects to Redis and
processes jobs from the 'downloads' queue. In Docker Compose this script is
invoked by a separate container so that long running downloads do not block
the API server.
"""

import os
import redis
from rq import Worker, Queue, Connection

if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    conn = redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker([Queue("downloads")])
        worker.work()