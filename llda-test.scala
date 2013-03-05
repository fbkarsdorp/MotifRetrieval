// Stanford TMT Example 3 - LDA inference on a new dataset
// http://nlp.stanford.edu/software/tmt/0.4/

// tells Scala where to find the TMT classes
import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

// the path of the model to load
if (args.length != 2) {
  System.err.println("Arguments: modelPath input.csv");
  System.err.println("  modelPath:  trained LLDA model");
  System.err.println("  input.csv:  path to input file with three comma separated columns: id, words, labels");
  System.exit(-1);
}

val modelPath = file(args(0));
println("Loading "+modelPath);
val lldamodel = LoadCVB0LabeledLDA(modelPath);
val model = lldamodel.asCVB0LDA;
// Or, for a Gibbs model, use:
// val model = LoadGibbsLDA(modelPath);

// A new dataset for inference.
val source = CSVFile(args(1)) ~> IDColumn(1);

val text = {
  source ~>                              // read from the source file
  Column(3) ~>                           // select column containing text
  TokenizeWith(model.tokenizer.get)      // tokenize with existing model's tokenizer
}

// Base name of output files to generate
val output = file(modelPath, source.meta[java.io.File].getName.replaceAll(".csv",""));

// turn the text into a dataset ready to be used with LDA
val dataset = LDADataset(text, termIndex = model.termIndex);

println("Writing document distributions to "+output+"-document-topic-distributions.csv");
val perDocTopicDistributions = InferCVB0DocumentTopicDistributions(model, dataset);
CSVFile(output+"-document-topic-distributuions.csv").write(perDocTopicDistributions);

// println("Writing topic usage to "+output+"-usage.csv");
// val usage = QueryTopicUsage(model, dataset, perDocTopicDistributions);
// CSVFile(output+"-usage.csv").write(usage);

// println("Estimating per-doc per-word topic distributions");
// val perDocWordTopicDistributions = EstimatePerWordTopicDistributions(
//   model, dataset, perDocTopicDistributions);
// //CSVFile(output+"-document-word-topic-distributions.csv").write(perDocWordTopicDistributions)

// println("Writing top terms to "+output+"-top-terms.csv");
// val topTerms = QueryTopTerms(model, dataset, perDocWordTopicDistributions, numTopTerms=50);
// CSVFile(output+"-top-terms.csv").write(topTerms);

// CSVFile(output+"per-doc-topic-assignment.csv").write({
//   for ((terms,(dId,dists)) <- text.iterator zip perDocWordTopicDistributions.iterator) yield {
//     require(terms.id == dId);
//     (terms.id,
//       for ((term,dist) <- (terms.value zip dists)) yield {
//      //for ((term,dist) <- (terms.value zip dists)) yield {
//        term + " " + dist.iterator.map({
//          case (topic,prob) => model.topicIndex.get.get(topic) + ":" + prob
//        }).mkString(" ");
//      });
//   }
// });

