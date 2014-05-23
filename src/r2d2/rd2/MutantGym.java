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

public class MutantGym implements Gym<AttributeVector, Integer>
{
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
