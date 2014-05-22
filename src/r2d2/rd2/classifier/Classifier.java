package r2d2.rd2.classifier;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Generic classifier template with data points of type D and class labels of
 * type C. A classifier must first be trained on a list of classifications
 * before it can classify new data points using #classify(dataPoint).
 */
public abstract class Classifier<D, C>
{
	/**
	 * Train the classifier on a training set
	 */
	public abstract void train(List<Classification<D, C>> trainingSet);

	/**
	 * Classify a data point
	 */
	public abstract C classify(D dataPoint);
	
	/**
	 * Classify a list of data points by repeatedly calling the
	 * implementation of #classify(dataPoint)
	 */
	public List<Classification<D, C>> classify(List<D> dataPoints)
	{
		return dataPoints.stream() // or parallelStream() 
				.map(d -> new Classification<D, C>(d, this.classify(d)))
				.collect(Collectors.toList());
	}
}
