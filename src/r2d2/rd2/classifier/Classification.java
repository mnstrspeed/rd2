package r2d2.rd2.classifier;

public class Classification<D, C>
{
	private D dataPoint;
	private C classLabel;

	public Classification(D dataPoint, C classLabel)
	{
		this.setDataPoint(dataPoint);
		this.setClassLabel(classLabel);
	}

	public D getDataPoint()
	{
		return this.dataPoint;
	}

	public void setDataPoint(D dataPoint)
	{
		this.dataPoint = dataPoint;
	}

	public C getClassLabel()
	{
		return this.classLabel;
	}

	public void setClassLabel(C classLabel)
	{
		this.classLabel = classLabel;
	}
	
	@Override
	public String toString()
	{
		return String.format("%s -> %s", 
				this.dataPoint.toString(),
				this.classLabel.toString());
	}
}
