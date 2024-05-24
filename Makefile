r:
	python -m ochat.serving.openai_api_server --model openchat/openchat-3.6-8b-20240522 --tensor-parallel-size 2 --port 18888

k:
	./kill.sh

c:
	./ca.sh