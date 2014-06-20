package r2d2.rd2.classifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
		List<Classification<D, C>> prototypes = trainingSet;
		
		prototypes = removeNoise(trainingSet);
		prototypes = reduceInstances(prototypes);
		
		super.train(prototypes);
	}

	private List<Classification<D, C>> removeNoise(List<Classification<D, C>> trainingSet)
	{
		List<Classification<D, C>> prototypes = new ArrayList<Classification<D, C>>();
		
		// Editing Nearest Neighbor (Wilson)
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
		return prototypes;
	}

	private List<Classification<D, C>> reduceInstances(List<Classification<D, C>> set)
	{
		List<Classification<D, C>> prototypes = new ArrayList<Classification<D, C>>(set);
		// DROP-3
		
		HashMap<Classification<D, C>, List<Classification<D, C>>> associates = 
				new HashMap<Classification<D, C>, List<Classification<D, C>>>(prototypes.size());
		HashMap<Classification<D, C>, List<Map.Entry<Classification<D, C>, Double>>> neighbors = 
				new HashMap<Classification<D, C>, List<Map.Entry<Classification<D, C>, Double>>>(prototypes.size());
		for (Classification<D, C> p : prototypes)
			associates.put(p, new LinkedList<Classification<D, C>>());
		
		// Add p to each of its neighbors' lists of associates
		for (Classification<D, C> p : prototypes)
		{
			List<Map.Entry<Classification<D, C>, Double>> pClosestNeighbors = 
					getClosestNeighbors(prototypes, this.distanceMeasure, p.getDataPoint(), this.k + 1);
			neighbors.put(p, pClosestNeighbors);
			for (Map.Entry<Classification<D, C>, Double> a : pClosestNeighbors)
			{
				associates.get(a.getKey()).add(p);
			}
		}
		
		// TODO: sort by distance to nearest enemy (descending)
		
		Iterator<Classification<D, C>> iterator = prototypes.iterator();
		while (iterator.hasNext())
		{
			Classification<D, C> p = iterator.next();
			
			// associates of p classified correctly with P as a neighbor
			int with = 0;
			for (Classification<D, C> a : associates.get(p))
				if (a.getClassLabel() == this.getMajority(neighbors.get(a)))
					with++;
			// associates of p classified correctly without P as a neighbor
			int without = 0;			
			for (Classification<D, C> a : associates.get(p))
				if (a.getClassLabel() == this.getMajority(withoutNeighbor(neighbors.get(a), p)))
					without++;
			
			if (without >= with)
			{
				//System.out.println("with: " + with + ", without: " + without + " (" + neighbors.get(p).size() + " associates)");
				iterator.remove(); // remove p from s
				
				for (Classification<D, C> a : associates.get(p))
				{
					// remove p from a's list of nearest neighbors & find a new nearest neighbor for a
					neighbors.put(a, getClosestNeighbors(prototypes, this.distanceMeasure, a.getDataPoint(), this.k + 1));
					// add a to its new neighbor's list of associates
					for (Map.Entry<Classification<D, C>, Double> n : neighbors.get(a))
					{
						if (!associates.get(n.getKey()).contains(a))
							associates.get(n.getKey()).add(a);
					}
				}
				for (Map.Entry<Classification<D, C>, Double> n : neighbors.get(p))
				{
					associates.get(n.getKey()).remove(p); // remove p from n's list of associates
				}
			}
		}
		
		return prototypes;
	}
	
	private static <T> List<Map.Entry<T, Double>> withoutNeighbor(List<Map.Entry<T, Double>> neighbors, T neighbor)
	{
		List<Map.Entry<T, Double>> result = new LinkedList<Map.Entry<T, Double>>(neighbors);
		Iterator<Map.Entry<T, Double>> iterator = result.iterator();
		while (iterator.hasNext())
		{
			if (iterator.next().getKey() == neighbor)
			{
				iterator.remove();
				break;
			}
		}
		return result;
	}
}
