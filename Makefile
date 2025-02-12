FILE_ID="1v4wByFuXsFvQjANY_Bi6lcmtNQocA25l"
FILE_NAME="deepfake_images.zip"
DATA_FOLDER="./data"

TIME=01:00:00

FRACTION=0.1
PARAMS = --fraction=$(FRACTION)\
	--nb_epochs=10
	
download-dataset:
	mkdir -p $(DATA_FOLDER)
	gdown --id $(FILE_ID) -O $(DATA_FOLDER)/$(FILE_NAME)
	unzip $(DATA_FOLDER)/$(FILE_NAME) -d $(DATA_FOLDER)
	rm $(DATA_FOLDER)/$(FILE_NAME)

make run:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py $(PARAMS)
