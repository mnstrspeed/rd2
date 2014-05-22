package r2d2.rd2.distances;

import r2d2.rd2.classifier.AttributeVector;

public class EuclideanDistance implements DistanceMeasure<AttributeVector>
{
	@Override
	public double compare(AttributeVector a, AttributeVector b)
	{
		double sum = 0;
		for (int i = 0; i < a.getDimension(); i++)
		{
			double d = a.get(i) - b.get(i);
			sum += d * d;
		}
		return Math.sqrt(sum);
	}
}
