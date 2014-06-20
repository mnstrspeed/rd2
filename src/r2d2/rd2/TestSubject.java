package r2d2.rd2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.Classifier;
import r2d2.rd2.classifier.KNearestNeighborClassifier;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.DistanceMeasure;
import r2d2.rd2.distances.EuclideanDistance;
import r2d2.rd2.distances.WeightedEuclideanDistance;
import r2d2.rd2.util.CrossValidation;

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
	
	public void normalizeChromosomes()
	{
		double prefactor = 0;
		double refactor = 0;
		
		for (int i = 0; i < chromosomes.length; i++)
		{
			if (chromosomes[i] > prefactor)
			{
				prefactor = chromosomes[i];
			}
		}
		
		for (int i = 0; i < chromosomes.length; i++)
		{
			chromosomes[i] = chromosomes[i] / prefactor;
		}
	}
	
	public void showChromosomes()
	{
		System.out.print("{");
		
		double total = 0;
		
		for (int i = 0; i < chromosomes.length - 1; i++)
		{
			System.out.print(chromosomes[i] + ", ");
		}
		System.out.print(chromosomes[chromosomes.length -1] + "} ");
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
		//ditness = fitness + diversity;
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
		normalizeChromosomes();
		fitness = determineFitness(chromosomes, set);
	}
	
	private double determineFitness(double[] child, List<Classification<AttributeVector, Integer>> set)
	{
		normalizeChromosomes();
		
		DistanceMeasure<AttributeVector> dist = new WeightedEuclideanDistance(child);
		Classifier<AttributeVector, Integer> classifier = 
				new KNearestNeighborClassifier<AttributeVector, Integer>(11, dist);
		
		int correct = 0, total = 0;
		for (CrossValidation.Set<Classification<AttributeVector, Integer>> crossValidationSet : 
			CrossValidation.KFold(set, 5))
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
