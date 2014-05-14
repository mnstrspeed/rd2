import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNearestNeighborClassifier<D, C> extends Classifier<D, C>
{
	private final int k;
	private final DistanceMeasure<D> distanceMeasure;
	
	private List<Classification<D, C>> trainingSet;
	
	public KNearestNeighborClassifier(final int k, final DistanceMeasure<D> distanceMeasure)
	{
		this.k = k;
		this.distanceMeasure = distanceMeasure;
	}
	
	@Override
	public void train(List<Classification<D, C>> trainingSet)
	{
		this.trainingSet = trainingSet;
	}

	@Override
	public C classify(D dataPoint)
	{
		if (this.trainingSet == null)
		{
			throw new NullPointerException("Classifier not trained yet");
		}
		
		List<Map.Entry<C, Double>> top = this.getClosestNeighbors(dataPoint);
		
		// count classes for first K points in training set
		HashMap<C, Integer> counts = new HashMap<C, Integer>(k);
		for (Map.Entry<C, Double> entry : top)
		{
			C c = entry.getKey();
			counts.put(c, counts.containsKey(c) ? counts.get(c) + 1 : 1);
		}
		
		// select majority class
		Map.Entry<C, Integer> majorityClass = null;
		for (Map.Entry<C, Integer> entry : counts.entrySet())
		{
			if (majorityClass == null || entry.getValue() > majorityClass.getValue())
				majorityClass = entry;
		}
		return majorityClass.getKey();
	}

	private List<Map.Entry<C, Double>> getClosestNeighbors(D dataPoint)
	{
		// modified max heap to find k closest data points in O(n log k)
		ArrayList<Map.Entry<C, Double>> heap = new ArrayList<Map.Entry<C, Double>>(this.k);
	
		for (Classification<D, C> c : this.trainingSet)
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

}
