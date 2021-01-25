use rand::{random, Rng};
use std::borrow::Borrow;
use std::ops::Deref;
use nalgebra::distance;
use ndarray::prelude::*;
use std::slice::{from_raw_parts};

fn sum_2d_vector(vector: &Vec<Vec<f64>>) -> f64
{
    let mut sum = 0.0;
    for i in 0..vector.len()
    {
        for j in 0..vector[0].len()
        {
            sum += vector[i][j];
        }
    }
    sum
}

fn mean(data: &Vec<f64>) -> Option<f64> {
    let sum = data.iter().sum::<f64>() as f64;
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f64),
        _ => None,
    }

}

fn std_deviation(data: &Vec<f64>) -> Option<f64> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - (*value as f64);

                diff * diff
            }).sum::<f64>() / count as f64;

            Some(variance.sqrt())
        },
        _ => None
    }
}


fn get_column_mean(cluster: &Vec<Vec<f64>>) -> Vec<f64>
{
    let mut result = Vec::new();

    for i in 0..cluster[0].len()
    {
        let mut sum = 0.0;
        for j in 0..cluster.len()
        {
            sum += cluster[j][i];
        }
        result.push(sum/cluster.len() as f64);
    }
    result
}

fn power(a: f64, power: i32) -> f64
{
    let mut result = a;
    for i in 1..power
    {
        result *= a;
    }
    result
}


fn get_distance(x1: &Vec<f64>, x2: &Vec<f64>) -> f64
{
    let mut sum = 0.0;

    for i in 0..x1.len()
    {

        sum += (x1[i] - x2[i]).powf(2.0);
    }

    sum.sqrt()
}

fn kmeans(X: &Vec<Vec<f64>>, k: i32, max_iters: i32) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)
{
    let mut cluster_list: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut converged = false;
    let mut current_iter = 0;
    let mut centroids:Vec<Vec<f64>> = Vec::new();

    for i in 0..k as usize
    {

        let mut rng = rand::thread_rng();
        let mut random_number = rng.gen_range(0, X.len()) as usize;

        centroids.push(X[random_number].clone());
    }



    while !converged && current_iter < max_iters
    {
        cluster_list = Vec::new();


        for i in 0..centroids.len()
        {
            let mut tmp: Vec<Vec<f64>> = Vec::new();
            cluster_list.push(tmp);
        }



        for x in X
        {
            let mut all_distances :Vec<f64>= Vec::new();

            for c in &centroids
            {
                all_distances.push(get_distance(x, c));
            }
            cluster_list[argmin(&all_distances) as usize].push(x.clone());
        }

        cluster_list.retain(|x| !x.is_empty() );

        let mut prev_centroids = centroids.clone();

        centroids = Vec::new();

        for j in 0..cluster_list.len()
        {
            centroids.push(get_column_mean(&cluster_list[j]));
        }

        let pattern: f64 =  (sum_2d_vector(&prev_centroids) - sum_2d_vector(&centroids)).abs();

        converged = (pattern == 0.0);

        current_iter += 1;
        break
    }

    //Standard deviation pas faite, peut être source de problème ?
    (centroids.clone(), cluster_list)

}

fn mat_T(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut aT: Vec<Vec<f64>> = Vec::new();
    for i in 0..a[0].len()
    {
        let mut row:Vec<f64> = Vec::new();
        for j in 0..a.len()
        {
            row.push(a[j][i]);
        }

        aT.push(row)
    }

    aT
}

fn mat_dot(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut dMat: Vec<Vec<f64>> = Vec::new();
    let mut bT: Vec<Vec<f64>> = mat_T(b);

    for i in 0..a.len()
    {
        let mut n_row: Vec<f64> = Vec::new();
        let mut row: Vec<f64> = a[i].clone();
        for j in 0..bT.len()
        {
            let mut col: Vec<f64> = bT[j].clone();
            let mut sum = 0.0;
            for k in 0..row.len()
            {
                sum += (row[k] * col[k]);
            }
            n_row.push(sum)
        }
        dMat.push(n_row);
    }

    dMat

}

fn inv(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>>
{
    let mut n = a.len();
    let mut b = vec![vec![0.0; n]; n];
    let mut p = vec![0.0; n];

    for i in 0..n
    {
        for j in 0..n
        {
            if i==j
            {
                b[i][j] = 1.0;

            }
            else
            {
                b[i][j] = 0.0;
            }
        }
    }

    for i in 0..n
    {
        p[i] = a[i][i];
        for j in 0..n
        {
            b[i][j] = b[i][j] / p[i];
            a[i][j] = a[i][j] / p[i];

        }

        for j in 0..n
        {
            for k in 0..n
            {
                if j != i
                {
                    p[j] = a[j][i];
                    b[j][k] -= b[i][k] * p[j];
                }
            }
        }

        for j in 0..n
        {
            for k in 0..n
            {
                if j != i
                {
                    a[j][k] -= a[i][k] * p[j];
                }
            }
        }
    }

    b
}

fn col(A: &Vec<Vec<f64>>, i: i32) -> Vec<Vec<f64>>
{
    let mut c:Vec<f64> = Vec::new();
    for j in 0..A.len()
    {
        c.push(A[j][i as usize]);
    }

    let mut n: Vec<Vec<f64>> = Vec::new();
    n.push(c);
    n

}

fn argmax(a: &Vec<f64>) -> f64
{
    let mut max_value = -f64::INFINITY;
    let mut max_index = 0;

    for i in 0..a.len()
    {
        if a[i] > max_value
        {
            max_value = a[i];
            max_index = i;
        }
    }

    max_index as f64
}

fn argmin(a: &Vec<f64>) -> f64
{
    let mut min_value = f64::INFINITY;
    let mut min_index = 0;

    for i in 0..a.len()
    {
        if a[i] < min_value
        {
            min_value = a[i];
            min_index = i;
        }
    }

    min_index as f64
}

pub struct RBF {

    X:Vec<Vec<f64>>,
    y:Vec<f64>,
    tX:Vec<Vec<f64>>,
    ty:Vec<f64>,
    num_of_classes: i32,
    k: i32,
    std_from_clusters: bool,
    Weights: Vec<Vec<f64>>,
    centroids: Vec<Vec<f64>>

}


impl RBF {
    pub fn new(new_X:Vec<Vec<f64>>, new_y:Vec<f64>, new_tX:Vec<Vec<f64>>, new_ty:Vec<f64>, new_num_of_classes: i32, new_k: i32, new_std_from_clusters: bool) -> RBF{
        RBF {
            X:new_X,
            y:new_y,
            tX:new_tX,
            ty:new_ty,
            num_of_classes: new_num_of_classes,
            k: new_k,
            std_from_clusters: new_std_from_clusters,
            Weights: Vec::new(),
            centroids: Vec::new()
        }

    }

    pub fn init(new_X:Vec<Vec<f64>>, new_y:Vec<f64>, new_tX:Vec<Vec<f64>>, new_ty:Vec<f64>, new_num_of_classes: i32, new_k: i32) -> RBF{
        RBF {
            X:new_X,
            y:new_y,
            tX:new_tX,
            ty:new_ty,
            num_of_classes: new_num_of_classes,
            k: new_k,
            std_from_clusters: false,
            Weights: Vec::new(),
            centroids: Vec::new()
        }

    }

    fn get_Acc(&self, x: &Vec<Vec<f64>>, y: &Vec<f64>, w: &Vec<Vec<f64>>, centroids: &Vec<Vec<f64>>, std: f64) -> f64
    {
        let ts_rbf_list = self.get_as_RBF_List(x, centroids, std);
        let pred_test_y_one_hot = mat_dot(&ts_rbf_list, w);

        let mut pred_test_y: Vec<f64> = Vec::new();
        for i in &pred_test_y_one_hot
        {
            pred_test_y.push(argmax(&i));
        }
        let mut true_counter = 0;

        for i in 0..pred_test_y.len()
        {
            if pred_test_y[i] == y[i]
            {
                true_counter+=1
            }
        }

        (true_counter / pred_test_y.len()) as f64

    }

    fn convert_to_one_hot(&self, x: &Vec<f64>, num_of_classes: i32) -> Vec<Vec<f64>>
    {
        let mut returned_result = vec![vec![0.0; num_of_classes as usize]; x.len()];
        for i in 0..x.len()
        {
            let c = x[i].clone();
            returned_result[i][c as usize] = 1.0
        }

        returned_result
    }

    fn rbf(&self, x: &Vec<f64>, c: &Vec<f64>, s: f64) -> f64
    {
        let distance = get_distance(x, c);
        1.0 / (-distance/power(s, 2)).exp()
    }

    fn get_as_RBF_List(&self, X: &Vec<Vec<f64>>, centroids: &Vec<Vec<f64>>, std: f64) -> Vec<Vec<f64>>
    {
        let mut rbf_list: Vec<Vec<f64>> = Vec::new();
        for x in rbf_list.clone()
        {
            let mut rbf_row: Vec<f64> = Vec::new();
            for c in centroids
            {
                rbf_row.push(self.rbf(&x, &c, std));
            }
            rbf_list.push(rbf_row);
        }

        rbf_list
    }

    pub fn fit(&mut self) -> f64
    {
        self.centroids = kmeans(&self.tX, self.k, 10).0;

        let mut dmax = 0.0;

        for i in &self.centroids
        {
            for j in self.tX.clone()
            {
                let d = get_distance(&i, &j);
                if d > dmax
                {
                    dmax = d;
                }
            }
        }
        let std = dmax / ((2*self.k) as f64).sqrt();

        let mut RBF_X = self.get_as_RBF_List(&self.tX, &self.centroids, std);

        let hot_tr_y = self.convert_to_one_hot(&self.ty, self.num_of_classes);

        /*let RBF_X_T = mat_T(&RBF_X);

        self.Weights = mat_dot(&mat_dot(&inv(mat_dot(&RBF_X_T, &RBF_X)), &RBF_X_T), &hot_tr_y);

        let mut accuracy = self.get_Acc(&self.tX, &self.ty, &self.Weights, &self.centroids, std);

        accuracy*/

        0.0
    }

    pub fn predict(&mut self, input_X: *mut f64, inputs_size: usize) -> i32
    {
        let mut input_X_vec;
        let mut dmax= 0.0;

        unsafe {

            input_X_vec = from_raw_parts(input_X, inputs_size).to_vec();
        }

        for i in &self.centroids
        {
            let d = get_distance(&i, &input_X_vec);
            if d > dmax
            {
                dmax = d;
            }
        }
        let std = dmax / ((2*self.k) as f64).sqrt();
        let input_X_2d = vec![input_X_vec; 1];

        let mut RBF_X = self.get_as_RBF_List(&input_X_2d, &self.centroids, std);

        let matrix_pred_value = mat_dot(&RBF_X, &self.Weights);

        let value_pred = argmax(&matrix_pred_value[0]);

        value_pred as i32

    }
}