package r2d2.rd2;

import java.util.List;

import r2d2.rd2.classifier.Classification;

public interface Gym<D, C>
{
	public void train(List<Classification<D, C>> train);
}
