package r2d2.rd2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.Classifier;
import r2d2.rd2.classifier.KNearestNeighborClassifier;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.DistanceMeasure;
import r2d2.rd2.distances.EuclideanDistance;
import r2d2.rd2.distances.WeightedEuclideanDistance;
import r2d2.rd2.util.CrossValidation;
import r2d2.rd2.util.StringHelper;

public class PlantsVsWeightsGym implements Gym<AttributeVector, Integer>
{
	public static class WeightVector implements Comparable<WeightVector>
	{
		public double[] weights;
		public double fitness;
		
		public WeightVector(double[] weights, double fitness)
		{
			this.weights = weights;
			this.fitness = fitness;
		}

		@Override
		public int compareTo(WeightVector b)
		{
			double d = this.fitness - b.fitness;
			if (d < 0)
				return 1;
			if (d > 0)
				return -1;
			return 0;
		}
		
		@Override
		public String toString()
		{
			return this.toWeightString() + " (" + this.fitness + ")";
		}
		
		public String toWeightString()
		{
			return "{ " + StringHelper.join(this.weights, ", ") + " }";
		}
	}
	
	private PriorityQueue<WeightVector> frontier;
	
	private List<CrossValidation.Set<Classification<AttributeVector, Integer>>> validationSets;
	private int k;
	
	public PlantsVsWeightsGym(int k)
	{
		this.k = k;
	}
	
	@Override
	public void train(List<Classification<AttributeVector, Integer>> set)
	{
		this.validationSets = CrossValidation.KFold(set, 5);
		
		int dimensions = set.get(0).getDataPoint().getDimension();
		WeightVector start = new WeightVector(
				new double[dimensions], 0); // initial weights (0, 0, ..., 0)
		
		this.frontier = new PriorityQueue<WeightVector>(1 << 16); // capacity of 2^16 
		frontier.add(start);
		
		double explored = 0;
		double nr = Math.pow((double)dimensions, 10.0);
		
		String previousLine = "";
		WeightVector best = start;
		while (!frontier.isEmpty())
		{
			WeightVector current = frontier.poll();
			
			System.out.print("\r");
			for (int i = 0; i < previousLine.length(); i++)
				System.out.print(" ");
			previousLine = Double.toString(explored / nr);
			System.out.print("\r" + previousLine);
			
			if (current.fitness > best.fitness)
			{
				best = current;
				System.out.println(" " + best);
				previousLine = "";
			}
			
			explored += dimensions;
			frontier.addAll(getChildren(current));
		}
	}
	
	private List<WeightVector> getChildren(WeightVector parent)
	{
		ArrayList<WeightVector> children = new ArrayList<WeightVector>(parent.weights.length);
		for (int i =0; i < parent.weights.length; i++)
		{
			double[] child = Arrays.copyOf(parent.weights, parent.weights.length);
			child[i] += 0.1;
			
			// keep weights between 0.0 and 1.0
			if (child[i] <= 1.0)
				children.add(new WeightVector(child, determineFitness(child)));
		}
		
		return children;
	}

	private double determineFitness(double[] child)
	{
		DistanceMeasure<AttributeVector> dist = new WeightedEuclideanDistance(child);
		Classifier<AttributeVector, Integer> classifier = new KNearestNeighborClassifier<AttributeVector, Integer>(this.k, dist);

		int correct = 0, total = 0;
		for (CrossValidation.Set<Classification<AttributeVector, Integer>> crossValidationSet : this.validationSets)
		{
			classifier.train(crossValidationSet.getTrainingSet());
			
			for (Classification<AttributeVector, Integer> testEntry : crossValidationSet.getTestSet())
			{
				if (classifier.classify(testEntry.getDataPoint()) == testEntry.getClassLabel())
					correct++;
				total++;
			}
		}
		
		return (correct / (double)total) * 100.0;
	}

}
