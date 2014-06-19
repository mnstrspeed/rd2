package r2d2.rd2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.EuclideanDistance;
import r2d2.rd2.distances.WeightedEuclideanDistance;

public class TestSubject {
	private double[] chromosomes;
	private double fitness;
	private double diversity;
	private double ditness;
	private Random destiny;
	
	public TestSubject(double[] dna, Random fate)
	{
		chromosomes = dna;
		fitness = 0;
		diversity = 0;
		ditness = fitness + diversity;
		destiny = fate;
	}
	
	public void setDiversity(double[] dna)
	{
		double total = 0;
		
		for (int i = 0; i < chromosomes.length; i++)
		{
			total = total + Math.sqrt((chromosomes[i] - dna[i]) * (chromosomes[i] - dna[i]));
		}
		
		diversity = total / chromosomes.length;
	}
	
	public void zeroDitness()
	{
		ditness = 0;
	}
	
	public void updateDitness()
	{
		ditness = fitness + diversity;
	}
	
	public double getFitness()
	{
		return fitness;
	}
	
	public double getDiversity()
	{
		return diversity;
	}
	
	public double getDitness()
	{
		return ditness;
	}
	
	public void setChromosome(int index, double value)
	{
		chromosomes[index] = value;
	}
	
	public double getChromosome(int index)
	{
		return chromosomes[index];
	}
	
	public double[] getdna()
	{
		return chromosomes;
	}
	
	public void setFitness(List<Classification<AttributeVector, Integer>> set)
	{
		fitness = determineFitness(chromosomes, set, destiny);
	}
	
	private static double determineFitness(double[] weights, List<Classification<AttributeVector, Integer>> set, Random random)
	{
		// Split set into train and test
		List<Classification<AttributeVector, Integer>> trainingSet = new ArrayList<Classification<AttributeVector, Integer>>();
		List<Classification<AttributeVector, Integer>> testSet = new ArrayList<Classification<AttributeVector, Integer>>();
		for (Classification<AttributeVector, Integer> c : set)
		{
			if (random.nextBoolean())
				trainingSet.add(c);
			else
				testSet.add(c);
		}
		
		// Train plain KNN classifier on trainingSet using weights
		R2D2Classifier<AttributeVector, Integer> classifier = new R2D2Classifier<AttributeVector, Integer>(11, 11,
				new WeightedEuclideanDistance(weights), 
				new EuclideanDistance());
		classifier.train(trainingSet);
		
		// Test performance on testSet
		int correct = 0;
		int total = 0;
		for (int i = 0; i < testSet.size(); i++)
		{
			Integer result = classifier.classify(testSet.get(i).getDataPoint());
			if (result == testSet.get(i).getClassLabel())
				correct++;
			total++;
		}
		
		// Return accuracy in %
		//System.out.println ((correct / (double)total) * 100.0);
		return (correct / (double)total) * 100.0;
	}
	
}
