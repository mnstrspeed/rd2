package r2d2.rd2.distances;

/**
 * Defines a distance measure between two instances of type T
 */
public interface DistanceMeasure<T>
{
	/**
	 * Compute the distance between a and b
	 */
	public double compare(T a, T b);
}
