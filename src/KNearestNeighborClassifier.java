import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

/**
 * A K-Nearest-Neighbor implementation
 */
public class KNearestNeighborClassifier<D, C> extends Classifier<D, C>
{
	protected final int k;
	protected final DistanceMeasure<D> distanceMeasure;
	
	protected List<Classification<D, C>> prototypes;
	
	public KNearestNeighborClassifier(final int k, final DistanceMeasure<D> distanceMeasure)
	{
		this.k = k;
		this.distanceMeasure = distanceMeasure;
	}
	
	@Override
	public void train(List<Classification<D, C>> trainingSet)
	{
		this.prototypes = trainingSet;
	}

	@Override
	public C classify(D dataPoint)
	{
		if (this.prototypes == null)
		{
			throw new NullPointerException("Classifier not trained yet");
		}
		
		List<Map.Entry<C, Double>> closestNeighbors = this.getClosestNeighbors(dataPoint);
		
		// Determine the majority class of the k closest neighbors
		C majority = null;
		int majorityCount = 0;
		for (Map.Entry<C, Double> entry : closestNeighbors)
		{
			C label = entry.getKey();
			if (majorityCount == 0)
			{
				majority = label;
				majorityCount = 1;
			}
			else if (majority.equals(label))
				majorityCount++;
			else
				majorityCount--;
		}

		return majority;
	}

	/**
	 * Finds the closest neighbors of #dataPoint in the training set.
	 */
	private List<Map.Entry<C, Double>> _getClosestNeighbors(D dataPoint)
	{
		LinkedList<Map.Entry<C, Double>> closest = new LinkedList<Map.Entry<C, Double>>(); // ordered max -> min, max k elements
		for (Classification<D, C> c : this.prototypes)
		{
			// Compute distance between this data point and the training instance
			double distance = distanceMeasure.compare(c.getDataPoint(), dataPoint);

			if (closest.size() < this.k || closest.get(0).getValue() > distance)
			{
				// Insert into the sorted list using insertion sort
				ListIterator<Map.Entry<C, Double>> iterator = closest.listIterator();
				while (iterator.hasNext())
				{
					if (iterator.next().getValue() < distance)
					{
						iterator.previous();
						iterator.add(new SimpleEntry<C, Double>(c.getClassLabel(), distance));
					}
				}
				// Trim off the first element (max) if our shortlist is too big now
				if (closest.size() > this.k)
				{
					closest.removeFirst();
				}
			}
		}
		return closest;
	}

	/**
	 * Finds the closest neighbors of #dataPoint in the training set.
	 */
	private List<Map.Entry<C, Double>> getClosestNeighbors(D dataPoint)
	{
		// modified max heap to find k closest data points in O(n log k)
		ArrayList<Map.Entry<C, Double>> heap = new ArrayList<Map.Entry<C, Double>>(this.k);
	
		for (Classification<D, C> c : this.prototypes)
		{
			double distance = distanceMeasure.compare(c.getDataPoint(), dataPoint);
			if (heap.size() < this.k)
			{
				heap.add(new SimpleEntry<C, Double>(c.getClassLabel(), distance));
				
				// swap up until heap property is satisfied again
				int pos = heap.size() - 1;
				while (pos > 0 && heap.get(pos).getValue() > heap.get((pos - 1) / 2).getValue())
				{
					Collections.swap(heap, pos, (pos - 1) / 2);
					pos = (pos - 1) / 2;
				}
			}
			else if (heap.get(0).getValue() > distance)
			{
				heap.set(0, new SimpleEntry<C, Double>(c.getClassLabel(), distance));
				
				// swap down until heap property is satisfied again
				int pos = 0;
				while (pos * 2 + 1 < this.k)
				{
					int largest = pos * 2 + 1;
					if (pos * 2 + 2 < this.k && heap.get(largest).getValue() < heap.get(pos * 2 + 2).getValue())
						largest = pos * 2 + 2;
						
					if (heap.get(pos).getValue() < heap.get(largest).getValue())
					{
						Collections.swap(heap, pos, largest);
						pos = largest;
					}
					else
					{
						break;
					}
				}
			}
		}
		// TODO: return List<C> without distances
		return heap;
	}

	/**
	 * Gets the prototypes resulting from training the classifier
	 */
	public List<Classification<D, C>> getPrototypes()
	{
		return this.prototypes;
	}
}
