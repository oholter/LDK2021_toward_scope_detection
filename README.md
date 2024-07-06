# LDK2021_toward_scope_detection in textual requirements
The tools used in the *toward scope detection in textual requirements* presented at LDK2021.

To reproduce the experiments:



## Get the documents
As of June 2024, the documents used in the paper can be downloaded from DNV at https://www.veracity.com/.
We used these standards:
* RU-SHIP
* ST-F101

## Extract PDF to XML

### Change variables
Change the variables in the main function in ``src/req_extract/src/main/java/req_extract/PDFParser.java``

* pdfPath = the path of your pdf-file
* outPath = path where you want the XML-file
* lastPage = last page number in document you want to read (or last page of the document)

### Compile and run the java code:
1. Install java and maven
2. Enter the pdf_parser folder
``cd src/req_extract``

3. Compile and execute
``mvn clean compile exec:java``

4. Confirm that the XML file is a valid XML. Note: You may have to manually correct mistakes in the XML file.




## Setup python environment

### Create and activate a virtual environment
Use *python 3.7.15*

``python -m venv venv``

``source venv/bin/activate``

### Install libraries
``python -m pip install -r requirements.txt``


## Extract sentences from XML

### Change variables
Change variables in main function in ``src/xmlparser/parsexml.py``

* path = the path of your XML-file
* tsv_path = the path of the output tsv-file
* extract sentences from the level you want in the document (uncomment the)

* Run the python file:

``python -m src.xmlparser.parsexml``


## Create gazetteers

### Create the Termostat lists
Transform the PDF document into a .txt document. This can be done, for example, with the ``pdftotext`` tool in Linux.
1. Input the text document into the [online termostat tool](http://termostat.ling.umontreal.ca/) and get the list of all the terms
2. Run ``python -m src.wn`` (change the input/output paths in ``src/wn.py``)

### Create the ISO 15926
1. Run ``python -m src.15926`` twice (once for each of the Artefact URIs) and collect the results
2. The list with Artefact CLASS must be curated according to the paper

### Create the list from word embeddings
* Run ``python -m src.vector_simil``


## Create labelled data
Run ``python -m src.snorkel_if_scope.labelling_functions``

1. check that the paths to the gazetteers in the beginning of the file are correct
2. check gold_path and requirements_path after ``if __name__`` ...
3. output filename is hardcoded toward the end of the file


## Train Bert-model
Run ``python -m src.bert_classifier.bert_train``. Uses command line arguments.

Example ``python -m src.bert_classifier.bert_train -e 10 --train /src/data.tsv --gold /src/gold.tsv --save model.bin --lr 3e-5 --full_finetuning --eps 1e-8``
will train a model using 10 epochs on data.tsv and evaluate with gold.tsv, save the model to model.bin, use a learning rate of 3e-5 and eps of 1e-8 and do fine-tuning on the bert-embeddings.


## Evaluate the model
Run ``python -m src.bert_classifier.bert_eval``. Use command line arguments ``--model and --test``.
