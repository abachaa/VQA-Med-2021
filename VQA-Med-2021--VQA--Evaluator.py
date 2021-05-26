import csv
import warnings
import nltk
import string
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
import tempfile
import json


# IMAGECLEF 2021 VQA-Med - VQA
class AIcrowdEvaluator:

  # ROUNDING OF SCORES
  ROUNDING_LIMIT = 4
  # Used for primiary score

  # Used for Bleu in NLTK (secondary score)
  remove_stopwords = True
  stemming = True
  case_sensitive = False

  def __init__(self, ground_truth_path, **kwargs):
    """
    This is the AIcrowd evaluator class which will be used for the evaluation.
    Please note that the class name should be `AIcrowdEvaluator`
    `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
    """
    # Ground truth file
    self.ground_truth_path = ground_truth_path
    # Ground truth annotations file
    self.gt_annotations_file_path = kwargs['gt_annotations_file_path']
    # Ground truth questions_file_path
    self.gt_questions_file_path = kwargs['gt_questions_file_path']
    # Load ground truth into memory
    self.gt, self.gt_image_ids_ordered = self.load_gt()
    # Load vqa 
    self.vqa = VQA(self.gt_annotations_file_path, self.gt_questions_file_path)
    

  def _evaluate(self, client_payload, _context={}):
    """
    This is the only method that will be called by the framework
    returns a _result_object that can contain up to 2 different scores
    `client_payload["submission_file_path"]` will hold the path of the submission file
    """
    print("evaluate...")
    # Load submission file path
    submission_file_path = client_payload["submission_file_path"]
    # Load preditctions and validate format
    predictions = self.load_predictions(submission_file_path)

    score = self.compute_primary_score(predictions)
    score_secondary = self.compute_secondary_score(predictions)

    _result_object = {
        "score": score,
        "score_secondary": score_secondary
    }
    
    assert "score" in _result_object
    assert "score_secondary" in _result_object

    return _result_object


  def load_gt(self):
    """
    Load and return groundtruth data
    """
    print("loading ground truth...")

    gt = {}
    gt_image_ids_ordered = []
    with open(self.ground_truth_path) as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            row  = line.split('|')
            image_id = row[0]
            category = "abnormality"
            question = "this is a place holder question"
            answers = []
            for ans in row [1:]:
               answers.append(ans.strip())
            # category = row[1]
            # question = row[2]
            # answer = row[3]
            gt[image_id] = (category, question, answers)
            gt_image_ids_ordered.append(image_id)

    return gt, gt_image_ids_ordered


  def load_predictions(self, submission_file_path):
    """
    Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
    """
    print("load predictions...")

    predictions = {}

    with open(submission_file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
        lineCnt = 0

        for row in reader:
            lineCnt += 1

            # Not 2 tokens on line => Error
            if len(row) != 2:
              self.raise_exception("Wrong format: Each line must consist of 2 tokens. " +
                "You have to specify an image ID followed by a pipe character (|) and " +
                "the answer string ({})", lineCnt, "<image_id>|<answer_string>")

            image_id = row[0]
            # Index out of bounds if more lines in submission file than in testset => Error
            try:
                expected_image_id = self.gt_image_ids_ordered[lineCnt - 1]
            except IndexError:
              self.raise_exception("Number of predictions greater then number of images in testset.",
                lineCnt)

            # Image ID not contained in testset => Error
            '''if image_id != expected_image_id:
              self.raise_exception("Unexpected image ID for line nbr {}. The expected ImageID is '{}'. " +
                "Please make sure to keep the same ordering of images as defined in the testset.",
                lineCnt, lineCnt, expected_image_id)'''
            
            if image_id not in self.gt_image_ids_ordered:
              self.raise_exception("Unexpected image ID for line nbr {}. This image ID does not exist in the test set.",
                lineCnt, lineCnt)
              
            answer = row[1].strip()
            if answer == "":
              self.raise_exception("Answer cannot be an empty string.", lineCnt)

            predictions[image_id] = answer

        if len(predictions) != len(self.gt):
          self.raise_exception("Number of predictions smaller than number of images in testset.",
            lineCnt)

    return predictions


  def raise_exception(self, message, record_count, *args):
    raise Exception(message.format(*args)+" Error occured at line nbr {}.".format(record_count))


  def compute_primary_score(self, predictions):
    """
    Compute and return the primary score
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Valiation should be handled in the load_predictions method
    """
    print("compute primary score...")

    predictions = self.convert_to_mscoco_format(predictions)

    vqaRes = self.vqa.loadRes(predictions, self.gt_questions_file_path)

    vqaEval = VQAEval(self.vqa, vqaRes, type(self).ROUNDING_LIMIT)
    vqaEval.evaluate()

    #print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))

    return vqaEval.accuracy['overall']
    #return vqaEval.accuracy


  def convert_to_mscoco_format(self, predictions):
    predictions_mscoco_format = []
    for key in predictions:
        local_answer = {}
        local_answer['question_id'] = key
        local_answer['answer'] = predictions[key]
        predictions_mscoco_format.append(local_answer)
    return predictions_mscoco_format


  def compute_secondary_score(self, predictions):
    """
    Compute and return the secondary score
    Ignore or remove this method if you do not have a secondary score to provide
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Valiation should be handled in the load_predictions method
    """
    print("compute secondary score...")

    return self.compute_bleu(predictions)


  def compute_bleu(self, predictions):
    # Hide warnings
    warnings.filterwarnings('ignore')

    # NLTK
    # Download Punkt tokenizer (for word_tokenize method)
    # Download stopwords (for stopword removal)
    try:
        nltk.data.find('tokenizers/punkt')
        stops = set(stopwords.words("english"))
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Remove punctuation from string
    translator = str.maketrans('', '', string.punctuation)

    # Define max score and current score
    max_score = len(self.gt)
    current_score = 0.0

    for image_key in predictions:
        candidate_caption = predictions[image_key]
        gt_caption = self.gt[image_key][2]
        bleu_score = 0.0
        if  len(gt_caption)==1:
            
            bleu_score = self.calc_single_blue_score(candidate_caption, gt_caption[0], self.gt, translator, stops,
                                                      stemmer)
        else:
           
            candidate_gt_captions = gt_caption
            bleu_scores_of_all_possibilities = []
            for gt_caption in candidate_gt_captions:
                bleu_scores_of_all_possibilities.append(
                    self.calc_single_blue_score(candidate_caption, gt_caption, self.gt, translator, stops,
                                                stemmer))
            bleu_score = max(bleu_scores_of_all_possibilities)

        # Increase calculated score
        current_score += bleu_score

    return round(current_score / max_score, type(self).ROUNDING_LIMIT)



  def calc_single_blue_score(self, candidate_caption, gt_caption, gt_pairs, translator, stops, stemmer):
    # Optional - Go to lowercase
    if not type(self).case_sensitive:
        candidate_caption = candidate_caption.lower()
        gt_caption = gt_caption.lower()

    # Split caption into individual words (remove punctuation)
    candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
    gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))

    # Optional - Remove stopwords
    if type(self).remove_stopwords:
        candidate_words = [word for word in candidate_words if word.lower() not in stops]
        gt_words = [word for word in gt_words if word.lower() not in stops]

    # Optional - Apply stemming
    if type(self).stemming:
        candidate_words = [stemmer.stem(word) for word in candidate_words]
        gt_words = [stemmer.stem(word) for word in gt_words]

    # Calculate BLEU score for the current caption
    try:
        # If both the GT and candidate are empty, assign a score of 1 for this caption
        if len(gt_words) == 0 and len(candidate_words) == 0:
            bleu_score = 1
        # Calculate the BLEU score
        else:
            bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words,
                                                                  smoothing_function=SmoothingFunction().method0)
    # Handle problematic cases where BLEU score calculation is impossible
    except ZeroDivisionError:
        pass
        # raise Exception('Problem with {} {}', gt_words, candidate_words)
    return bleu_score


import json
import datetime
import copy
#__author__ = 'aagrawal'
#__version__ = '0.9'
# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"
class VQA:
  def __init__(self, annotation_file=None, question_file=None):
      """
          Constructor of VQA helper class for reading and visualizing questions and answers.
      :param annotation_file (str): location of VQA annotation file
      :return:
      """
      # load dataset
      self.dataset = {}
      self.questions = {}
      self.qa = {}
      self.qqa = {}
      self.imgToQA = {}
      if not annotation_file == None and not question_file == None:
          #print('loading VQA annotations and questions into memory...')
          #time_t = datetime.datetime.utcnow()
          dataset = json.load(open(annotation_file, 'r'))
          questions = json.load(open(question_file, 'r'))
          #print(datetime.datetime.utcnow() - time_t)
          self.dataset = dataset
          self.questions = questions
          self.createIndex()

  def createIndex(self):
      # create index
      #print('creating index...')
      imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
      qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
      qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
      for ann in self.dataset['annotations']:
          imgToQA[ann['image_id']] += [ann]
          qa[ann['question_id']] = ann
      for ques in self.questions['questions']:
          qqa[ques['question_id']] = ques
      #print('index created!')

      # create class members
      self.qa = qa
      self.qqa = qqa
      self.imgToQA = imgToQA

  def info(self):
      """
      Print information about the VQA annotation file.
      :return:
      """
      for key, value in self.dataset['info'].items():
          print('%s: %s' % (key, value))

  def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
      """
      Get question ids that satisfy given filter conditions. default skips that filter
      :param 	imgIds    (int array)   : get question ids for given imgs
              quesTypes (str array)   : get question ids for given question types
              ansTypes  (str array)   : get question ids for given answer types
      :return:    ids   (int array)   : integer array of question ids
      """
      imgIds = imgIds if type(imgIds) == list else [imgIds]
      quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
      ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

      if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
          anns = self.dataset['annotations']
      else:
          if not len(imgIds) == 0:
              anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
          else:
              anns = self.dataset['annotations']
          anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
          anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
      ids = [ann['question_id'] for ann in anns]
      return ids

  def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
      """
      Get image ids that satisfy given filter conditions. default skips that filter
      :param quesIds   (int array)   : get image ids for given question ids
              quesTypes (str array)   : get image ids for given question types
              ansTypes  (str array)   : get image ids for given answer types
      :return: ids     (int array)   : integer array of image ids
      """
      quesIds = quesIds if type(quesIds) == list else [quesIds]
      quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
      ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

      if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
          anns = self.dataset['annotations']
      else:
          if not len(quesIds) == 0:
              anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])
          else:
              anns = self.dataset['annotations']
          anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
          anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
      ids = [ann['image_id'] for ann in anns]
      return ids

  def loadQA(self, ids=[]):
      """
      Load questions and answers with the specified question ids.
      :param ids (int array)       : integer ids specifying question ids
      :return: qa (object array)   : loaded qa objects
      """
      if type(ids) == list:
          return [self.qa[id] for id in ids]
      elif type(ids) == int:
          return [self.qa[ids]]
      else:
          return [self.qa[ids]]

  def showQA(self, anns):
      """
      Display the specified annotations.
      :param anns (array of object): annotations to display
      :return: None
      """
      if len(anns) == 0:
          return 0
      for ann in anns:
          quesId = ann['question_id']
          print("Question: %s" % (self.qqa[quesId]['question']))
          for ans in ann['answers']:
              print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

  def loadRes(self, anns, quesFile):
      """
      Load result file and return a result object.
      :param   resFile (str)     : file name of result file
      :return: res (obj)         : result api object
      """
      res = VQA()
      res.questions = json.load(open(quesFile))
      res.dataset['info'] = copy.deepcopy(self.questions['info'])
      res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
      res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
      res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
      res.dataset['license'] = copy.deepcopy(self.questions['license'])


      assert type(anns) == list, 'results is not an array of objects'
      annsQuesIds = [ann['question_id'] for ann in anns]
      assert set(annsQuesIds) == set(self.getQuesIds()), \
          'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
      for ann in anns:
          quesId = ann['question_id']
          if res.dataset['task_type'] == 'Multiple Choice':
              assert ann['answer'] in self.qqa[quesId][
                  'multiple_choices'], 'predicted answer is not one of the multiple choices'
          qaAnn = self.qa[quesId]
          ann['image_id'] = qaAnn['image_id']
          ann['question_type'] = qaAnn['question_type']
          ann['answer_type'] = qaAnn['answer_type']

      res.dataset['annotations'] = anns
      res.createIndex()
      return res


import sys
import re
#__author__ = 'aagrawal'
# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
# Adapted for ImageCLEF VQA-Med 2021 by Vivek Datla, Sadid A. Hasan, and Mourad Sarrouti.

class VQAEval: 
  def __init__(self, vqa, vqaRes, n=2):
      self.n = n
      self.accuracy = {}
      self.evalQA = {}
      self.evalQuesType = {}
      self.evalAnsType = {}
      self.vqa = vqa
      self.vqaRes = vqaRes
      self.params = {'question_id': vqa.getQuesIds()}
      self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                            "couldnt": "couldn't", \
                            "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                            "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                            "hed": "he'd", "hed've": "he'd've", \
                            "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                            "Id've": "I'd've", "I'dve": "I'd've", \
                            "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                            "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                            "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                            "mightn'tve": "mightn't've", "mightve": "might've", \
                            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                            "oclock": "o'clock", "oughtnt": "oughtn't", \
                            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                            "shed've": "she'd've", "she'dve": "she'd've", \
                            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                            "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                            "someone'dve": "someone'd've", \
                            "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                            "somethingd've": "something'd've", \
                            "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                            "thered": "there'd", "thered've": "there'd've", \
                            "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                            "theyd've": "they'd've", \
                            "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                            "twas": "'twas", "wasnt": "wasn't", \
                            "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                            "whatll": "what'll", "whatre": "what're", \
                            "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                            "wheres": "where's", "whereve": "where've", \
                            "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                            "whos": "who's", "whove": "who've", "whyll": "why'll", \
                            "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                            "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                            "yall'd've": "y'all'd've", \
                            "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                            "youd've": "you'd've", "you'dve": "you'd've", \
                            "youll": "you'll", "youre": "you're", "youve": "you've"}
      self.manualMap = {'none': '0',
                        'zero': '0',
                        'one': '1',
                        'two': '2',
                        'three': '3',
                        'four': '4',
                        'five': '5',
                        'six': '6',
                        'seven': '7',
                        'eight': '8',
                        'nine': '9',
                        'ten': '10'
                        }
      self.articles = ['a',
                        'an',
                        'the'
                        ]

      self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
      self.commaStrip = re.compile("(\d)(\,)(\d)")
      self.punct = [';', r"/", '[', ']', '"', '{', '}',
                    '(', ')', '=', '+', '\\', '_', '-',
                    '>', '<', '@', '`', ',', '?', '!']

  ###
  # Since the annotations are performed by single annoatator for each instance, we do not need the user's
  # response to have atleast 3 matches. We changed this part of the code.

  ###

  def evaluate(self, quesIds=None):
      if quesIds == None:
          quesIds = [quesId for quesId in self.params['question_id']]
      # quesIds = quesIds[:10000]
      gts = {}
      res = {}
      for quesId in quesIds:
          gts[quesId] = self.vqa.qa[quesId]
          res[quesId] = self.vqaRes.qa[quesId]

      # =================================================
      # Compute accuracy
      # =================================================
      accQA = []
      accQuesType = {}
      accAnsType = {}

      for quesId in quesIds:
          resAns = res[quesId]['answer']
          resAns = resAns.replace('\n', ' ')
          resAns = resAns.replace('\t', ' ')
          resAns = resAns.strip()
          resAns = self.processPunctuation(resAns)
          resAns = self.processDigitArticle(resAns)
          gtAcc = []

          for ansDic in gts[quesId]['answers']:
              ansDic['answer'] = self.processPunctuation(ansDic['answer'])
              ansDic['answer'] = self.processDigitArticle(ansDic['answer'])

          otherGTAns = [item for item in gts[quesId]['answers']]
          matchingAns = [item for item in otherGTAns if item['answer'] == resAns]
          acc = min(1, float(len(matchingAns)))
          gtAcc.append(acc)


          quesType = gts[quesId]['question_type']
          ansType = gts[quesId]['answer_type']
          avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
          accQA.append(avgGTAcc)
          if quesType not in accQuesType:
              accQuesType[quesType] = []
          accQuesType[quesType].append(avgGTAcc)
          if ansType not in accAnsType:
              accAnsType[ansType] = []
          accAnsType[ansType].append(avgGTAcc)
          self.setEvalQA(quesId, avgGTAcc)
          self.setEvalQuesType(quesId, quesType, avgGTAcc)
          self.setEvalAnsType(quesId, ansType, avgGTAcc)


      self.setAccuracy(accQA, accQuesType, accAnsType)

  def processPunctuation(self, inText):
      outText = inText
      for p in self.punct:
          if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
              outText = outText.replace(p, '')
          else:
              outText = outText.replace(p, ' ')
      outText = self.periodStrip.sub("",
                                      outText,
                                      re.UNICODE)
      return outText

  def processDigitArticle(self, inText):
      outText = []
      tempText = inText.lower().split()
      for word in tempText:
          word = self.manualMap.setdefault(word, word)
          if word not in self.articles:
              outText.append(word)
          else:
              pass
      for wordId, word in enumerate(outText):
          if word in self.contractions:
              outText[wordId] = self.contractions[word]
      outText = ' '.join(outText)
      return outText
  '''
  # We want to use these functions later for analytics
  def setAccuracy(self, accQA, accQuesType, accAnsType):
      self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)
      self.accuracy['perQuestionType'] = {
      quesType: round(100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]), self.n) for quesType in
      accQuesType}
      self.accuracy['perAnswerType'] = {
      ansType: round(100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n) for ansType in
      accAnsType}

  def setEvalQA(self, quesId, acc):
      self.evalQA[quesId] = round(100 * acc, self.n)

  def setEvalQuesType(self, quesId, quesType, acc):
      if quesType not in self.evalQuesType:
          self.evalQuesType[quesType] = {}
      self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

  def setEvalAnsType(self, quesId, ansType, acc):
      if ansType not in self.evalAnsType:
          self.evalAnsType[ansType] = {}
      self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)
  
  
  '''


  def setAccuracy(self, accQA, accQuesType, accAnsType):
      self.accuracy['overall'] = round(float(sum(accQA)) / len(accQA), self.n)
      self.accuracy['perQuestionType'] = {
      quesType: round(float(sum(accQuesType[quesType])) / len(accQuesType[quesType]), self.n) for quesType in
      accQuesType}
      self.accuracy['perAnswerType'] = {
      ansType: round(float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n) for ansType in
      accAnsType}

  def setEvalQA(self, quesId, acc):
      self.evalQA[quesId] = round(acc, self.n)

  def setEvalQuesType(self, quesId, quesType, acc):
      if quesType not in self.evalQuesType:
          self.evalQuesType[quesType] = {}
      self.evalQuesType[quesType][quesId] = round(acc, self.n)

  def setEvalAnsType(self, quesId, ansType, acc):
      if ansType not in self.evalAnsType:
          self.evalAnsType[ansType] = {}
      self.evalAnsType[ansType][quesId] = round(acc, self.n)

  def updateProgress(self, progress):
      barLength = 20
      status = ""
      if isinstance(progress, int):
          progress = float(progress)
      if not isinstance(progress, float):
          progress = 0
          status = "error: progress var must be float\r\n"
      if progress < 0:
          progress = 0
          status = "Halt...\r\n"
      if progress >= 1:
          progress = 1
          status = "Done...\r\n"
      block = int(round(barLength * progress))
      text = "\rFinshed Percent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), int(progress * 100),
                                                        status)
      sys.stdout.write(text)
      sys.stdout.flush()




# TEST THIS EVALUATOR
if __name__ == "__main__":

    # Path of file that contains ground truth
    ground_truth_path = "data/resources/Task1-VQAnswering2021-Test-ReferenceAnswers-4Evaluator.txt"
    # Path of file containing gt answers in mscoco format
    gt_annotations_file_path = "data/resources/Task1-VQAnswering2021-Test-ReferenceAnswers_mscoco_format_vqa.json"
    # Path of file containing gt questions in mscoco format
    gt_questions_file_path = "data/resources/Task1-VQAnswering2021-Test-ReferenceQuestions_mscoco_format_vqa.json"

    #submission_file_path = "data/test_runs/not_perfect.txt"
    submission_file_path = "data/test_runs/perfect.txt"
    #submission_file_path = "data/test_runs/01_not_2_tokens_on_line.txt"
    # submission_file_path = "data/test_runs/02_wrong_image_id.txt"
    # submission_file_path = "data/test_runs/03_empty_answer.txt"
    # submission_file_path = "data/test_runs/04_image_id_more_than_once.txt"
    # submission_file_path = "data/test_runs/05_too_many_images.txt"
    # submission_file_path = "data/test_runs/06_not_all_images_included.txt"
    

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    
    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator and pass non-standard arguments as kwargs
    aicrowd_evaluator = AIcrowdEvaluator(ground_truth_path, gt_annotations_file_path=gt_annotations_file_path,
      gt_questions_file_path=gt_questions_file_path)
    
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
