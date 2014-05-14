import java.util.List;
import java.util.stream.Collectors;

public abstract class Classifier<D, C>
{
	public abstract void train(List<Classification<D, C>> trainingSet);
	public abstract C classify(D dataPoint);
	
	public List<Classification<D, C>> classify(List<D> dataPoints)
	{
		return dataPoints.parallelStream()
				.map(d -> new Classification<D, C>(d, this.classify(d)))
				.collect(Collectors.toList());
	}
}
