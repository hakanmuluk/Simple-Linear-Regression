import java.util.ArrayList;

public class LinearRegression{
    double[] w;
    double b;
    double[][] x;
    double[] y;
    int m;
    int noOfFeatures;
    double learningRate;
    public LinearRegression(double[] w, double b, double[][] x, double[] y, double learningRate){
        this.w = w;
        this.b = b;
        this.x = x;
        this.y = y;
        m = x.length;
        noOfFeatures = x[0].length;
        this.learningRate = learningRate;
    }
    public double dotProduct(double[] w, double[] x){
        double total = w[0] * x[0];
        for(int i = 1; i < w.length; i++){
            total += w[i] * x[i];
        }
        return total;
    }
    public double computeCost(){
        double cost = 0;
        for(int i = 0; i < m; i++){
            //cost += dotProduct(w, x[i]);    
            cost += Math.pow(dotProduct(w, x[i]) + b - y[i], 2);
        }
        return cost / (2 * m);
    }
    public double[] computeGradients(){
        double[] gradients = new double[w.length + 1];//Last one is b's gradient
        double djdb = 0;
        for(int i = 0; i < noOfFeatures; i++){
            double djdw = 0;
            for(int j = 0; j < m; j++){
                djdw = djdw + (dotProduct(w, x[j]) + b - y[j]) * x[j][i];
                if(i == 0)
                    djdb = djdb + (dotProduct(w, x[j]) + b - y[j]);
            }
            
            gradients[i] = djdw;
        }
        gradients[gradients.length - 1] = djdb;
        return gradients;
    }
    public void scaleFeatures(){ // Scale by average, (-1,1)
        for(int i = 0; i < noOfFeatures; i++){
            double total = 0;
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for(int j = 0; j < m; j++){
                total += x[j][i];
                if(x[j][i] < min)
                    min = x[j][i];
                if(x[j][i] > max)
                    max = x[j][i];
            }
            for(int j = 0; j < m; j++){
                x[j][i] = (x[j][i] - (total / m)) / (max - min); 
            }
        }
    }
    public void doLinearRegression(){
        scaleFeatures();
        double lastCost = computeCost();
        int noOfIter = 0;
        while(noOfIter < 100000){
            double presentCost = computeCost();
            if(noOfIter >= 1 && Math.abs(presentCost - lastCost) < 0.000000000000000000000000000001){
                System.out.print("Converged to " + presentCost + ", ");
                for(int i = 0; i < w.length; i++){
                    System.out.print("w" + (i + 1) + " = " + w[i]);
                }
                System.out.print(", b = " + b);
                break;
            }
            else if(noOfIter >= 1){
                if(lastCost >= presentCost)
                    lastCost = presentCost;
                else{
                    System.out.println("Diverges");
                //break;
                }
            }
            double[] gradients = computeGradients();
            for(int i = 0; i < gradients.length - 1; i++){
                w[i] = w[i] - learningRate * gradients[i];
            }
            b = b - learningRate * gradients[gradients.length - 1];
            if(noOfIter % 100 == 0){
            for(int i = 0; i < w.length; i++){
                System.out.print("w" + (i + 1) + " = " + w[i]);
            }
            System.out.println(", b = " + b);
        }
            noOfIter++;
        }
    }

}