import java.util.ArrayList;
import java.util.List;


public class R2D2Classifier<D, C> extends KNearestNeighborClassifier<D, C>
{
	public R2D2Classifier(int k, DistanceMeasure<D> distanceMeasure)
	{
		super(k, distanceMeasure);
	}

	@Override
	public void train(List<Classification<D, C>> trainingSet)
	{
		List<Classification<D, C>> prototypes = 
				new ArrayList<Classification<D, C>>();
		
		// Prepare KNN classifier for ENN
		KNearestNeighborClassifier<D, C> classifier = 
				new KNearestNeighborClassifier<D, C>(this.k + 1, this.distanceMeasure);
		classifier.train(trainingSet);
		
		for (Classification<D, C> c : trainingSet)
		{
			if (classifier.classify(c.getDataPoint()).equals(c.getClassLabel()))
			{
				prototypes.add(c);
			}
		}
		
		super.train(prototypes);
	}
}
