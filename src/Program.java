import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Function;

public class Program
{
	public static void main(String[] args)
	{
		long time = System.currentTimeMillis();
		
		List<Classification<AttributeVector, Integer>> trainingSet;
		List<AttributeVector> testSet;
		
		String trainSetPath = args[0];
		String testSetPath = args[1];
		
		// Load
		try
		{
			trainingSet = read(trainSetPath, scanner -> 
				new Classification<AttributeVector, Integer>(
					AttributeVector.fromScanner(scanner, 8), scanner.nextInt()));
			System.out.println("Done loading " + trainSetPath);
			
			testSet = read(testSetPath, scanner -> AttributeVector.fromScanner(scanner, 8));
			System.out.println("Done loading " + testSetPath);
		}
		catch (IOException ex)
		{
			throw new RuntimeException("SYSTEM FAILURE SYSTEM FAILURE SYSTEM FAILURE", ex);
		}
		
		String prototypeOutputPath = args[2];
		String labelOutputPath = args[3];
		classify(trainingSet, testSet, prototypeOutputPath, labelOutputPath);
		//trainWeights(trainingSet);
		
		System.out.println("Execution finished in " + (System.currentTimeMillis() - time) + " ms");
	}
	
	private static void trainWeights(List<Classification<AttributeVector, Integer>> set)
	{
		List<Classification<AttributeVector, Integer>> prototypes = 
				new ArrayList<Classification<AttributeVector, Integer>>();
		
		// Prepare KNN classifier for ENN
		KNearestNeighborClassifier<AttributeVector, Integer> classifier = 
				new KNearestNeighborClassifier<AttributeVector, Integer>(12, EUCLIDEAN_DISTANCE);
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
		
		// Determine initial fitness (with all weights at 0.5)
		double fitness = determineFitness(chromosomes, trainingSet, testSet);
		System.out.println("initial: " + StringHelper.join(chromosomes) + " (" + fitness + ")");
		
		// Initialize random stream
		Random random = new Random(48151623);
		while (true)
		{
			// Mutate chromosomes
			double[] mutated = mutate(chromosomes, random);
			// Determine fitness of mutated chromosomes
	        double mutatedFitness = determineFitness(mutated, trainingSet, testSet);
	        
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
	
	private static double determineFitness(double[] weights, 
			List<Classification<AttributeVector, Integer>> trainingSet,
			List<Classification<AttributeVector, Integer>> testSet)
	{
		// Train plain KNN classifier on trainingSet using weights
		KNearestNeighborClassifier<AttributeVector, Integer> classifier =
				new KNearestNeighborClassifier<AttributeVector, Integer>(11, new WeightedEuclidean(weights));
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

	private static void classify(List<Classification<AttributeVector, Integer>> trainingSet,
			List<AttributeVector> testSet, String prototypeOutputPath, String labelOutputPath)
	{
		// Weights to use
		WeightedEuclidean weightedEuclideanDistance = new WeightedEuclidean(new double[] {
				0.08405771704168705, 0.6819710657563258, 0.32148339074121934, 0.4721965196198296, 
				0.6700739421381516, 0.5174025332034238, 0.4088143948627616, 0.40369326223664026
		});
		
		// Prepare classifier using training set
		R2D2Classifier<AttributeVector, Integer> classifier = 
			new R2D2Classifier<AttributeVector, Integer>(11, weightedEuclideanDistance, EUCLIDEAN_DISTANCE);
		classifier.train(trainingSet);

		// Classify test set
		List<Classification<AttributeVector, Integer>> predicted = classifier.classify(testSet);
		
		// Save prototypes and class labels for upload
		try
		{
			write(prototypeOutputPath, classifier.getPrototypes(), c ->
					c.getDataPoint() + " " + c.getClassLabel());
			write(labelOutputPath, predicted, c -> c.getClassLabel().toString());
		}
		catch (IOException ex)
		{
			throw new RuntimeException(ex);
		}
	}
	
	private static final DistanceMeasure<AttributeVector> EUCLIDEAN_DISTANCE =
			(AttributeVector a, AttributeVector b) ->
	{
		double sum = 0;
		for (int i = 0; i < a.getDimension(); i++)
		{
			double d = a.get(i) - b.get(i);
			sum += d * d;
		}
		return Math.sqrt(sum);
	};
	
	public static class WeightedEuclidean implements DistanceMeasure<AttributeVector>
	{
		private double[] weights;
		
		public WeightedEuclidean(double[] weights)
		{
			this.weights = weights;
		}
		
		@Override
		public double compare(AttributeVector a, AttributeVector b)
		{
			double sum = 0;
			for (int i = 0; i < a.getDimension(); i++)
			{
				double d = a.get(i) - b.get(i);
				sum += weights[i] * (d * d);
			}
			return sum;
		}
	}
	
	/**
	 * Read a number of T instances from a file using a Function<Scanner, T> to
	 * read individual instances
	 */
	private static <T> List<T> read(String filePath, Function<Scanner, T> reader) throws FileNotFoundException
	{
		List<T> set = new ArrayList<T>();
		
		Scanner scanner = new Scanner(new FileInputStream(filePath));
		while (scanner.hasNext())
		{
			set.add(reader.apply(scanner));
		}
		scanner.close();
		return set;
	}
	
	private static <T> void write(String filePath, List<T> elements, Function<T, String> writer) throws IOException
	{
		try (PrintWriter output = new PrintWriter(new FileWriter(filePath)))
		{
			for (T element : elements)
			{
				output.println(writer.apply(element));
			}
		}
	}
}
