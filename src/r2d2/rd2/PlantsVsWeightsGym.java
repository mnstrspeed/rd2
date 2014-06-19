package r2d2.rd2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.Classifier;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.DistanceMeasure;
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
	
	private List<Classification<AttributeVector, Integer>> set;
	
	private int classifierK;
	private int noiseK;
	
	public PlantsVsWeightsGym(int classifierK, int noiseK)
	{
		this.classifierK = classifierK;
		this.noiseK = noiseK;
	}
	
	@Override
	public void train(List<Classification<AttributeVector, Integer>> set)
	{
		this.set = set;
		
		int dimensions = set.get(0).getDataPoint().getDimension();
		WeightVector start = new WeightVector(
				new double[dimensions], 0); // initial weights (0, 0, ..., 0)
		
		this.frontier = new PriorityQueue<WeightVector>(1 << 16); // capacity of 2^16 
		frontier.add(start);
		
		double explored = 0;
		double nr = Math.pow((double)dimensions, 10.0);
		
		WeightVector best = start;
		while (!frontier.isEmpty())
		{
			WeightVector current = frontier.poll();
			//System.out.println("(" + (explored / nr) + ") " + current);
			
			if (current.fitness > best.fitness)
			{
				best = current;
				System.out.println("(" + (explored / nr) + ") " + best);
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
			
			// cap at 1.0
			if (child[i] <= 1.0)
				children.add(new WeightVector(child, determineFitness(child)));
		}
		
		return children;
	}

	private double determineFitness(double[] child)
	{
		DistanceMeasure<AttributeVector> dist = new WeightedEuclideanDistance(child);
		R2D2Classifier<AttributeVector, Integer> classifier = 
				new R2D2Classifier<AttributeVector, Integer>(classifierK, noiseK, dist, dist);
		
		int correct = 0, total = 0;
		for (CrossValidation.Set<Classification<AttributeVector, Integer>> crossValidationSet : 
			CrossValidation.KFold(this.set, 5))
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

	/*
	public void trainWithStats(List<Classification<AttributeVector, Integer>> trainingSet,
			List<AttributeVector> testSet, List<Integer> testLabelSet)
	{
		this.train = trainingSet.subList(0, trainingSet.size() / 2);
		this.test = trainingSet.subList(trainingSet.size() / 2, trainingSet.size());
		
		int dimensions = train.get(0).getDataPoint().getDimension();
		WeightVector start = new WeightVector(
				new double[dimensions], 0); // initial weights (0, 0, ..., 0)
		
		this.frontier = new PriorityQueue<WeightVector>(1 << 16); // capacity of 2^16 
		frontier.add(start);
		
		int j = 0;
		
		WeightVector best = start;
		while (!frontier.isEmpty())
		{
			WeightVector current = frontier.poll();
			
			if (current.fitness >= best.fitness)
			{
				best = current;
				//System.out.println(best);
				
				DistanceMeasure<AttributeVector> dist = new WeightedEuclideanDistance(best.weights);
				R2D2Classifier<AttributeVector, Integer> classifier = 
						new R2D2Classifier<AttributeVector, Integer>(classifierK, noiseK, dist, dist);
				classifier.train(this.train);
				
				int correct = 0;
				int total = 0;
				for (int i = 0; i < testSet.size(); i++)
				{
					Integer result = classifier.classify(testSet.get(i));
					if (result == testLabelSet.get(i))
						correct++;
					total++;
				}
				
				double accuracy = (correct / (double)total) * 100.0;
				System.out.println(j + " " + best.fitness + " " + accuracy);
			}
			
			frontier.addAll(getChildren(current));
			j++;
		}
	}*/
}
