package r2d2.rd2.classifier;
import java.util.ArrayList;
import java.util.List;

import r2d2.rd2.distances.DistanceMeasure;


public class R2D2Classifier<D, C> extends KNearestNeighborClassifier<D, C>
{
	private int ennK;
	private DistanceMeasure<D> ennDistanceMeasure;
	
	public R2D2Classifier(int k, int ennK, DistanceMeasure<D> distanceMeasure, DistanceMeasure<D> ennDistanceMeasure)
	{
		super(k, distanceMeasure);
		this.ennK = ennK;
		this.ennDistanceMeasure = ennDistanceMeasure;
	}

	@Override
	public void train(List<Classification<D, C>> trainingSet)
	{
		List<Classification<D, C>> prototypes = 
				new ArrayList<Classification<D, C>>();
		
		// Prepare KNN classifier for ENN
		KNearestNeighborClassifier<D, C> classifier = 
				new KNearestNeighborClassifier<D, C>(this.ennK, this.ennDistanceMeasure);
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
