package r2d2.rd2.distances;

import r2d2.rd2.classifier.AttributeVector;

public class WeightedEuclideanDistance implements DistanceMeasure<AttributeVector>
{
	private double[] weights;
	
	public WeightedEuclideanDistance(double[] weights)
	{
		this.weights = weights;
	}
	
	@Override
	public double compare(AttributeVector a, AttributeVector b)
	{
		double sum = 0;
		for (int i = 0; i < a.getDimension(); i++)
		{
			double d = a.get(i) - b.get(i);
			sum += weights[i] * (d * d);
		}
		return sum;
	}
}
