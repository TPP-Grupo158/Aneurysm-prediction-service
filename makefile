build:
	@echo "Activating venv and exporting path..."
	. .venv/bin/activate && \
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd)/Intracranial_Aneurysm_Detection/ && \
	fastapi dev main.py
run:
	&& \
	
	python path_setup.py 