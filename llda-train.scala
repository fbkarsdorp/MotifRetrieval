// Stanford TMT Example 6 - Training a LabeledLDA model
// http://nlp.stanford.edu/software/tmt/0.3/

// tells Scala where to find the TMT classes
import scalanlp.io._;
import java.io.{File,FileWriter};
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;
import scalanlp.collection.LazyIterable;
import scalanlp.util.TopK;
import java.io._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;
import edu.stanford.nlp.tmt.learn._;
import edu.stanford.nlp.tmt.model._;

println(args.length)
if (args.length != 4) {
  System.err.println("Arguments: input.csv alpha beta iterations");
  System.exit(-1);
}

val source = CSVFile(args(0)) ~> IDColumn(1);

val tokenizer = {
  //SimpleEnglishTokenizer() ~>            // tokenize on space and punctuation
  WhitespaceTokenizer() ~>               // use Whitespace tokenizer as Ucto does a better job ;-)
  CaseFolder() ~>                        // lowercase everything
  WordsAndNumbersOnlyFilter() ~>         // ignore non-words and non-numbers
  MinimumLengthFilter(1)                 // take terms with >=3 characters
}

val text = {
  source ~>                              // read from the source file
  Column(3) ~>                           // select column containing text
  TokenizeWith(tokenizer) ~>             // tokenize with tokenizer above
  TermCounter() ~>                      // collect counts (needed below)
  TermMinimumDocumentCountFilter(1) ~>  // filter terms in <4 docs
  // TermStopListFilter(List("de", "en", "van", "ik", "te", "dat", "die", "in", "een", 
  //   "hij", "het", "niet", "zijn", "is", "was", "op", "aan", "met", "als", "voor", 
  //   "had", "er", "maar", "om", "hem", "dan", "zou", "of", "wat", "mijn", "men", "dit", 
  //   "zo", "door", "over", "ze", "zich", "bij", "ook", "tot", "je", "mij", "uit", "der", 
  //   "daar", "haar", "naar", "heb", "hoe", "heeft", "hebben", "deze", "u", "want", "nog", 
  //   "zal", "me", "zij", "nu", "ge", "geen", "omdat", "iets", "worden", "toch", "al", 
  //   "waren", "veel", "meer", "doen", "toen", "moet", "ben", "zonder", "kan", "hun", 
  //   "dus", "alles", "onder", "ja", "eens", "hier", "wie", "werd", "altijd", "doch", 
  //   "wordt", "wezen", "kunnen", "ons", "zelf", "tegen", "na", "reeds", "wil", "kon", 
  //   "niets", "uw", "iemand", "geweest", "andere")) ~>
  TermDynamicStopListFilter(0) ~>       // filter out 30 most common terms
  DocumentMinimumLengthFilter(1)         // take only docs with >=5 terms
}

// define fields from the dataset we are going to slice against
val labels = {
  source ~>                              // read from the source file
  Column(2) ~>                           // take column two, the motifs
  TokenizeWith(WhitespaceTokenizer()) ~> // turns label field into an array
  TermCounter() ~>                         // collect label counts
  TermMinimumDocumentCountFilter(1)    // WATCH THIS!!! filter labels in < 10 docs?
}

val dataset = LabeledLDADataset(text, labels);

// define the model parameters
val modelParams = LabeledLDAModelParams(dataset, args(1).toDouble, args(2).toDouble);

// Name of the output model folder to generate
val modelPath = file("llda-vvb-"+dataset.signature+"-"+modelParams.signature);

// Trains the model, writing to the given output path
TrainCVB0LabeledLDA(modelParams, dataset, output = modelPath, maxIterations = args(3).toInt);
//TrainGibbsLabeledLDA(modelParams, dataset, output = modelPath, maxIterations = 1500);

val f = new PrintWriter(new File(args(0) + ".config"));
f.print(modelPath)
f.close()

// val model = LoadCVB0LabeledLDA(modelPath);
// val topTerms = QueryTopTerms(model, model.numTerms);
// CSVFile(modelPath+"/top-terms.csv").write(topTerms);

