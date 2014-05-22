package r2d2.rd2.classifier;
import java.util.ArrayList;
import java.util.List;

import r2d2.rd2.distances.DistanceMeasure;


public class R2D2Classifier<D, C> extends KNearestNeighborClassifier<D, C>
{
	private DistanceMeasure<D> ennDistanceMeasure;
	
	public R2D2Classifier(int k, DistanceMeasure<D> distanceMeasure, DistanceMeasure<D> ennDistanceMeasure)
	{
		super(k, distanceMeasure);
		this.ennDistanceMeasure = ennDistanceMeasure;
		//System.out.println("Initializing classifier with K=" + this.k);
	}

	@Override
	public void train(List<Classification<D, C>> trainingSet)
	{
		//System.out.println("Filtering noise with ENN, K=" + this.k);
		
		List<Classification<D, C>> prototypes = 
				new ArrayList<Classification<D, C>>();
		
		// Prepare KNN classifier for ENN
		KNearestNeighborClassifier<D, C> classifier = 
				new KNearestNeighborClassifier<D, C>(this.k + 1, this.ennDistanceMeasure);
		classifier.train(trainingSet);
		
		for (Classification<D, C> c : trainingSet)
		{
			if (classifier.classify(c.getDataPoint()).equals(c.getClassLabel()))
			{
				prototypes.add(c);
			}
		}
		//System.out.println("Discarded " + ((trainingSet.size() - prototypes.size()) / 
		//		(float)trainingSet.size() * 100) + "% as noise");
		
		super.train(prototypes);
	}
}
