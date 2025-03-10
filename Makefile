FILE_ID="1v4wByFuXsFvQjANY_Bi6lcmtNQocA25l"
FILE_NAME="deepfake_images.zip"
DATA_FOLDER="./data"

PART=ENSTA-h100 #ENSTA-h100 #ENSTA-l40s
TIME=04:00:00

FRACTION=1.0

# BATCH_SIZES=20 32 64 128 256
# NB_EPOCHS=20 50 100
# LEARNING_RATES=1e-5

BATCH_SIZES=64
LEARNING_RATES=1e-5

# Try 1e-4 after

PARAMS = --fraction=$(FRACTION)\
	--nb_epochs=50\
	--batch_size=20\
	--lr=1e-5

download-dataset:
	mkdir -p $(DATA_FOLDER)
	gdown --id $(FILE_ID) -O $(DATA_FOLDER)/$(FILE_NAME)
	unzip $(DATA_FOLDER)/$(FILE_NAME) -d $(DATA_FOLDER)
	rm $(DATA_FOLDER)/$(FILE_NAME)

make run:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py $(PARAMS)

make run-meso:
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python MesoNet-Pytorch/train_Meso.py -n 'Mesonet' -tp "data/Train" -vp "data/Test" -bz 10 -e 50 -mn 'meso4.pkl' 

run-grid-search:
	@for batch_size in $(BATCH_SIZES); do \
	  for nb_epochs in $(NB_EPOCHS); do \
	    for lr in $(LEARNING_RATES); do \
	      echo "Running with batch_size=$$batch_size, nb_epochs=$$nb_epochs, lr=$$lr"; \
	      srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py \
	      --fraction=$(FRACTION) --nb_epochs=$$nb_epochs --batch_size=$$batch_size --lr=$$lr; \
	    done; \
	  done; \
	done

run-grid-search-2:
	@for batch_size in $(BATCH_SIZES); do \
	    for lr in $(LEARNING_RATES); do \
	      echo "Running with batch_size=$$batch_size, nb_epochs=30, lr=$$lr"; \
	      srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python main.py \
	      --fraction=$(FRACTION) --nb_epochs=30 --batch_size=$$batch_size --lr=$$lr; \
	    done; \
	done