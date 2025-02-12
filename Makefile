

PART=ENSTA-h100 #ENSTA-h100 #ENSTA-l40s
TIME=01:00:00

DATA_PATH="./data"
FRACTION=0.1
BATCH_SIZE=64

PARAMS = --data_path=$(DATA_PATH)\
	--fraction=$(FRACTION)\
	--batch_size=$(BATCH_SIZE)


setup:

	pip install -r requirements.txt

	@printf "\033[92msetup done\033[0m\n"
	


run:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py $(PARAMS)

