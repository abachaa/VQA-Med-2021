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


# IMAGECLEF 2021 VQA-Med - VQG
#  Code written by Vivek Datla, Sadid A. Hasan, and Mourad Sarrouti.

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
    self.ground_truth_path = ground_truth_path
    self.gt, self.gt_image_ids_ordered = self.load_gt()
    

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

    bleu_score = self.compute_primary_score(predictions)
    score_secondary = self.compute_secondary_score(predictions)

    _result_object = {
        "score": bleu_score,
        "score_secondary": 0
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
        reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
        for row in reader:
            image_id = row[0]
            questions = [i.strip() for i in row[1:] ] 
            #print(questions)
            gt[image_id] = (questions)
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

            # Less than 2 tokens on line => Error
            if len(row) < 2:
              self.raise_exception("Wrong format: Each line must consist of at least 2 tokens. " +
                "You have to specify an image ID followed by a pipe character (|) and " +
                "the question string ({}).", lineCnt, "<image_id>|<question>")

            image_id = row[0]
            # Index out of bounds if more lines in submission file than in testset => Error
            try:
                expected_image_id = self.gt_image_ids_ordered[lineCnt - 1]
            except IndexError:
              self.raise_exception("Number of predictions greater then number of images in testset.",
                lineCnt)

            # Image ID not contained in testset => Error
            if image_id not in self.gt_image_ids_ordered:
              self.raise_exception("Unexpected image ID for line nbr {}. This image ID does not exist in the test set.",
                lineCnt, lineCnt)

            questions = [i.strip() for i in row[1:]]
            
            questions_text = " ".join(row[1:]).strip()
            if questions_text == "":
                self.raise_exception("Question cannot be an empty string.", lineCnt)

            predictions[image_id] = questions

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

    return self.compute_bleu(predictions)


  def compute_secondary_score(self, predictions):
    """
    Compute and return the secondary score
    Ignore or remove this method if you do not have a secondary score to provide
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Valiation should be handled in the load_predictions method
    """
    print("compute secondary score...")

    return 0.0


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
        candidate_captions = predictions[image_key]
        gt_captions = self.gt[image_key]
        bleu_score = 0.0
        if  len (gt_captions) == 1 and len(candidate_captions) ==1 :
            
            bleu_score = self.calc_single_blue_score(candidate_captions[0], gt_captions[0], self.gt, translator, stops,
                                                      stemmer)
        else:
            
            candidate_gt_captions = gt_captions
            bleu_scores_of_all_possibilities = []
            for gt_caption in candidate_gt_captions:
                for candidate_caption in candidate_captions:
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


# TEST THIS EVALUATOR
if __name__ == "__main__":

    ground_truth_path = "data/resources/Task2-VQGeneration2021-Test-ReferenceQuestions_vqg.txt"

    #submission_file_path = "data/test_runs/00_01_RUN_NOT_PERFECT_VQG.txt"
    submission_file_path = "data/test_runs/00_02_RUN_PERFECT_VQG.txt"
    #submission_file_path = "data/test_runs/01_less_than_2_tokens_on_line.txt"
    #submission_file_path = "data/test_runs/02_wrong_image_id.txt"
    #submission_file_path = "data/test_runs/03_empty_question.txt"
    #submission_file_path = "data/test_runs/04_image_id_more_than_once.txt"
    # submission_file_path = "data/test_runs/05_too_many_images.txt"
    # submission_file_path = "data/test_runs/06_not_all_images_included.txt"

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path
    
    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator
    aicrowd_evaluator = AIcrowdEvaluator(ground_truth_path)
    
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
