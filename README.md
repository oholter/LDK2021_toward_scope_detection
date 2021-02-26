# LDK2021_toward_scope_detection in textual requirements
The tools used in the *toward scope detection in textual requirements* (will be) submitted to LDK2021

To reproduce the experiments:

## Install the libraries in the requirements.txt-file
(pip install -r requirements.txt)

## Get the requirements
We used these standards:
* [RU-SHIP](https://rules.dnvgl.com/ServiceDocuments/dnvgl/#!/industry/1/Maritime/1/DNV%20GL%20rules%20for%20classification:%20Ships%20(RU-SHIP)) (openly available)
* [ST-F101](https://oilgas.standards.dnvgl.com/download/dnvgl-st-f101-submarine-pipeline-systems) (subscription required)


## Extract pdf to xml
* Run ``src/req_extract/src/main/java/req_extract/PDFParser.java`` (e.g., open the project in eclipse and add maven dependencies it depends on Apache PDFbox)
* change variables in main function:
    * pdfPath = the path of your pdf-file
    * outPath = path where you want the XML-file
    * lastPage = last page number in document you want to read (or last page of the document)

## Extract sentences from XML
* Run ``src/xmlparser/parsexml.py``
* Change variables in main function:
    * path = the path of your XML-file
    * tsv_path = the path of the output tsv-file
    * extract sentences from the level you want in the document (uncomment the)

## Create gazetteers
### Create the Termostat lists
1. Input the document into the [online termostat tool](http://termostat.ling.umontreal.ca/) and get the list of all the terms
2. Run src/wn.py (change the path in read-csv)
3. Copy the list to a text-file

### Create the ISO 15926
1. Run ``15926.py`` twice (once for each of the Artefact URIs) and collect the results
2. The list with Artefact CLASS must be curated according to the paper

### Create the list from word embeddings
* Run ``src/vector_simil.py``


## Create labelled data
Run ``src/snorkel_if_scope/labelling_functions.py``
1. check that the paths to the gazetteers in the beginning of the file are correct
2. check gold_path and requirements_path after ``if __name__`` ...
3. output filename is hardcoded toward the end of the file


## Train Bert-model
Run ``src/bert_classifier/bert_train.py``. Uses command line arguments.
Example ``python -e 10 --train /src/data.tsv --gold /src/gold.tsv --save model.bin --lr 3e-5 --full_finetuning --eps 1e-8``
will train a model using 10 epochs on data.tsv and evaluate with gold.tsv, save the model to model.bin, use a learning rate of 3e-5 and eps of 1e-8 and do fine-tuning on the bert-embeddings.


## Evaluate the model
Run ``src/bert_classifier/bert_eval.py``. Uses command line arguments --model and --test.
