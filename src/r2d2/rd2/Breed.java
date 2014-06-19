package r2d2.rd2;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.KNearestNeighborClassifier;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.EuclideanDistance;
import r2d2.rd2.distances.WeightedEuclideanDistance;
import r2d2.rd2.util.StringHelper;

public class Breed implements Gym<AttributeVector, Integer>
{
	
	// Initialize random stream
	Random random = new Random(48151623);
	
	TestSubject[] parents = new TestSubject[20];
	TestSubject[] children = new TestSubject[800];

	public void train(List<Classification<AttributeVector, Integer>> set)
	{
			
		genesis();
		System.out.println("test");
		
		while (true)
		{
			breed();
			select(set);
			
			parents[0].setFitness(set);
			System.out.println(parents[0].getFitness());
		}
	}
	
	public void genesis()
	{
		double[] baseX = new double[8];
		for (int i = 0; i < baseX.length; i++)
		{
			baseX[i] = 0;
		}
		
		double[] baseY = new double[8];
		for (int i = 0; i < baseY.length; i++)
		{
			baseY[i] = 1;
		}
		
		double[] baseXY = new double[8];
		for (int i = 0; i < baseXY.length; i++)
		{
			baseXY[i] = i % 2;
		}
		
		double[] baseYX = new double[8];
		for (int i = 0; i < baseYX.length; i++)
		{
			baseYX[i] = (i + 1) % 2;
		}
		
		for (int i = 0; i < 8; i++)
		{	
			parents[i] = new TestSubject(baseX, random);
			parents[i].setChromosome(i, 1);
		}
		
		for (int i = 8; i < 16; i++)
		{	
			parents[i] = new TestSubject(baseY, random);
			parents[i].setChromosome(i - 8, 1);
		}
		
		parents[16] = new TestSubject(baseX, random);
		parents[17] = new TestSubject(baseY, random );
		parents[18] = new TestSubject(baseXY, random);
		parents[19] = new TestSubject(baseYX, random);
		
	}
	
	public void breed()
	{
		int index = 0;
		
		for (int i = 0; i < parents.length; i++)
		{
			for (int j = 0; j < parents.length; j++)
			{
				double[] dna = new double[8];
				for (int k = 0; k < dna.length; k++)
				{
					dna[k] = (parents[i].getChromosome(k) + parents[j].getChromosome(k)) / 2;
				}
				children[index]= new TestSubject(dna, random);
				index++;
			}
		}
		
		for (int i = 0; i < parents.length; i++)
		{
			for (int j = 0; j < parents.length; j++)
			{
				double[] dna = new double[8];
				for (int k = 0; k < dna.length; k++)
				{
					dna[k] = (parents[i].getChromosome(k) + parents[j].getChromosome(k)) / 2;
				}
				
				dna = mutate(dna, random);			
				children[index]= new TestSubject(dna, random);
				index++;
			}
		}
	}
	
	public void select(List<Classification<AttributeVector, Integer>> set)
	{
		double maxFitness = 0;
		int fittestSubject = 400;
		double maxDiversity = 0;
		int mostDivergent = 400;
		
		for (int i = 0; i < children.length; i++)
		{
			children[i].setFitness(set);
			//System.out.println(children[i].getFitness());
		}
		
		for (int i = 0; i < children.length; i++)
		{
			if (children[i].getFitness() > maxFitness)
			{
				maxFitness = children[i].getFitness();
				fittestSubject = i;
				//System.out.println(i);
			}
		}
		
		double[] fittestdna = new double[8];
		for (int i = 0; i < fittestdna.length; i++)
		{
			//System.out.println(children[fittestSubject].getChromosome(i));
			fittestdna[i] = children[fittestSubject].getChromosome(i);
		}
		
		parents[0] = new TestSubject(fittestdna, random);
		
		for (int i = 0; i < children.length; i++)
		{
			children[i].setDiversity(fittestdna);
		}
		
		for (int i = 0; i < children.length; i++)
		{
			if (children[i].getDiversity() > maxDiversity)
			{
				maxDiversity = children[i].getDiversity();
				mostDivergent = i;
			}
		}
		
		double[] mostdivergentdna = new double[8];
		for (int i = 0; i < mostdivergentdna.length; i++)
		{
			mostdivergentdna[i] = children[mostDivergent].getChromosome(i);
		}
		
		parents[1] = new TestSubject(mostdivergentdna, random);
		
		for (int i = 0; i < children.length; i++)
		{
			children[i].updateDitness();
		}
		
		children[fittestSubject].zeroDitness();
		children[mostDivergent].zeroDitness();

		for (int i = 2; i < 16; i++)
		{
			double maxDitness = 0;
			int mostDitnesst = 400;
			
			for (int j = 0; j < children.length; j ++)
			{
				if (children[j].getDitness() > maxDitness)
				{
					maxDitness = children[j].getDitness();
					mostDitnesst = j;
				}
			}
			
			parents[i] = new TestSubject(children[mostDitnesst].getdna(), random);
			
		}
		
	}

	/*
	public void train(List<Classification<AttributeVector, Integer>> set)
	{
		List<Classification<AttributeVector, Integer>> prototypes = 
				new ArrayList<Classification<AttributeVector, Integer>>();
		
		// Prepare KNN classifier for ENN
		KNearestNeighborClassifier<AttributeVector, Integer> classifier = 
				new KNearestNeighborClassifier<AttributeVector, Integer>(12, new EuclideanDistance());
		classifier.train(set);
		
		for (Classification<AttributeVector, Integer> c : set)
		{
			if (classifier.classify(c.getDataPoint()).equals(c.getClassLabel()))
			{
				prototypes.add(c);
			}
		}
		
		List<Classification<AttributeVector, Integer>> trainingSet = prototypes.subList(0, prototypes.size() / 2);
		List<Classification<AttributeVector, Integer>> testSet = prototypes.subList(prototypes.size() / 2, prototypes.size());
		
		double[] chromosomes = new double[8];
		for (int i = 0; i < chromosomes.length; i++)
			chromosomes[i] = 0.5;
		
		// Initialize random stream
		Random random = new Random(48151623);
		
		// Determine initial fitness (with all weights at 0.5)
		double fitness = determineFitness(chromosomes, set, random);
		System.out.println("initial: " + StringHelper.join(chromosomes) + " (" + fitness + ")");
		
		while (true)
		{
			// Mutate chromosomes
			double[] mutated = mutate(chromosomes, random);
			// Determine fitness of mutated chromosomes
	        double mutatedFitness = determineFitness(mutated, set, random);
	        
	        // Keep mutations if they improved the fitness
	        if (mutatedFitness > fitness)
	        {
	        	chromosomes = mutated;
	        	fitness = mutatedFitness;
	        	
	        	System.out.println("better: " + StringHelper.join(chromosomes) + " (" + fitness + ")");
	        }
		}
	}
	
	*/
	
	private static double[] mutate(double[] chromosomes, Random random)
	{
		double[] mutated = Arrays.copyOf(chromosomes, chromosomes.length);
		
		// Make a random number of mutations
		int numberOfMutations = random.nextInt(mutated.length);
		for (int i = 0; i < numberOfMutations; i++)
		{
			// Mutate a random weight with +/- 0.10
			int pointMutation = random.nextInt(mutated.length);
			mutated[pointMutation] = mutated[pointMutation] + (random.nextDouble() * 0.20 - 0.10);
		}
		
        return mutated;
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
		return (correct / (double)total) * 100.0;
	}
}
