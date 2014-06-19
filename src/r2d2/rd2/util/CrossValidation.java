package r2d2.rd2.util;

import java.util.ArrayList;
import java.util.List;

public class CrossValidation
{
	public static class Set<T>
	{
		private List<T> trainingSet;
		private List<T> testSet;
		
		public Set(List<T> trainingSet, List<T> testSet)
		{
			this.trainingSet = trainingSet;
			this.testSet = testSet;
		}
		
		public List<T> getTrainingSet()
		{
			return this.trainingSet;
		}
		
		public List<T> getTestSet()
		{
			return this.testSet;
		}
	}
	
	public static <T> List<CrossValidation.Set<T>> KFold(List<T> set, int k)
	{
		ArrayList<CrossValidation.Set<T>> crossValidationSets = new ArrayList<CrossValidation.Set<T>>(k);
		
		// Divide set into k partitions
		List<List<T>> partitions = new ArrayList<List<T>>(k);
		int n = set.size() / k;
		for (int i = 0; i < k; i++)
		{
			partitions.add(set.subList(i * n, (i + 1) * n));
		}
		
		// Compile cross validation sets
		for (int i = 0; i < k; i++)
		{
			List<T> testSet = partitions.get(i);
			List<T> trainingSet = new ArrayList<T>();
			for (int j = 0; j < k; j++)
			{
				if (i != j)
					trainingSet.addAll(partitions.get(j));
			}
			crossValidationSets.add(new CrossValidation.Set<T>(testSet, trainingSet));
		}
		
		return crossValidationSets;
	}
	
	public static <T> List<CrossValidation.Set<T>> LeaveOneOut(List<T> set)
	{
		return CrossValidation.KFold(set, set.size());
	}
}
