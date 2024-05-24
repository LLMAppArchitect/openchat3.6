#!/bin/bash
ps -ef|grep -E 'ochat.serving.openai_api_server|ray'|awk '{print $2}'| xargs kill -9