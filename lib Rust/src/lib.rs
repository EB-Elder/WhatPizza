use std::slice::{from_raw_parts};
use rand::Rng;


fn power(a: f64, power: i32) -> f64
{
    let mut result = a;
    for i in 0..power
    {
        result *= a;
    }
    result
}

fn sign(value: f64) -> f64
{
    if value >= 0.0
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}

#[no_mangle]
pub extern fn create_linear_model(input_size: usize) -> *mut Vec<f64> {

    //Initialisation du model lineaire avec une taille défini + un biais
    let mut weights = Vec::with_capacity(input_size + 1);

    //Initialisation des poids
    for i in 0..input_size {
        weights.push(rand::thread_rng().gen_range(-1.0, 1.0));

    }
    //Initialisation du poids du biais
    weights.push(rand::thread_rng().gen_range(-1.0, 1.0));


    //Fuite mémoire volontaire afin de pouvoir renvoyer un pointeur
    let boxed_weights = Box::new(weights);
    let boxed_ref = Box::leak(boxed_weights);

    boxed_ref
}

#[no_mangle]
pub extern fn create_mlp_model(input_size: usize, number_hidden_layer: usize, neurones_count: *mut i32) -> *mut Vec<f64> {


    //TODO: CREATION DE TABLEAU 2D
    //Initialisation du model lineaire avec une taille défini + un biais

    let mut neurones_count_slice;
    let mut final_capacity:usize = input_size +  1;
    unsafe {
        neurones_count_slice = from_raw_parts(neurones_count, number_hidden_layer);
    }


    for i in 0..number_hidden_layer{
        final_capacity+=neurones_count_slice[i] as usize + 1;
    }

    //Ajout de toutes les neurones et des biais
    let mut weights = Vec::with_capacity(final_capacity);
    //Initialisation des poids
    for i in 0..final_capacity {
        weights.push(rand::thread_rng().gen_range(-1.0, 1.0));

    }


    //Fuite mémoire volontaire afin de pouvoir renvoyer un pointeur
    let boxed_weights = Box::new(weights);
    let boxed_ref = Box::leak(boxed_weights);

    boxed_ref

}

#[no_mangle]
pub extern fn get_weights(model: *mut Vec<f64>, index: usize) -> f64
{
    let boxed_model;
    unsafe {
        boxed_model = model.as_ref().unwrap();
    }
    boxed_model[index]
}

#[no_mangle]
pub extern fn predict_mlp_model_classification(model: *mut Vec<f64>,
                                                  inputs: *mut f64, inputs_size: usize, number_hidden_layer: usize, neurones_count: *mut i32) -> f64 {
    let boxed_model;
    let inputs_slice;
    let neurones_count_slice;
    let mut neurones_values:Vec<Vec<f64>> = Vec::with_capacity(number_hidden_layer);


    unsafe {
        //Récupération des contenus des pointeurs
        boxed_model = model.as_ref().unwrap();
        inputs_slice = from_raw_parts(inputs, inputs_size);
        neurones_count_slice = from_raw_parts(neurones_count, number_hidden_layer);
    }

    for i in 0..neurones_count_slice.len(){
        let mut tmp= Vec::with_capacity(i);
        for k in 0..i{
            tmp.push(0.0);
        }
        neurones_values.push(tmp);
    }

    let mut result = 0.0;
    let biais = 1.0;

    //prédictions

    for HL in 0..number_hidden_layer
    {
        if HL == 0
        {
            for neurones in 0..neurones_count_slice[0]
            {
                for entree in 0..inputs_size
                {
                    neurones_values[HL][neurones as usize] += boxed_model[0 * number_hidden_layer + entree] * inputs_slice[entree];
                }
                neurones_values[HL][neurones as usize] += boxed_model[0 * number_hidden_layer + neurones_count_slice[0] as usize] * biais;
                neurones_values[HL][neurones as usize] = sign(neurones_values[HL][neurones as usize]);

            }

        }
        else if HL == number_hidden_layer-1
        {
            for entree in 0..neurones_count_slice[HL]
            {
                result += boxed_model[HL * number_hidden_layer + entree as usize] * neurones_values[HL as usize][entree as usize];
            }
            result += boxed_model[HL * number_hidden_layer + neurones_count_slice[HL] as usize] * biais;
            result = sign(result);
        }
        else
        {
            for neurones in 0..neurones_count_slice[HL]
            {
                for entree in 0..neurones_count_slice[HL-1]
                {
                    neurones_values[HL][neurones as usize] += boxed_model[(HL-1) * number_hidden_layer + entree as usize] * neurones_values[HL-1][entree as usize];
                }

                neurones_values[HL][neurones as usize] += boxed_model[HL * number_hidden_layer + neurones_count_slice[number_hidden_layer] as usize] * biais;
                neurones_values[HL][neurones as usize] = sign(neurones_values[HL][neurones as usize]);
            }

        }

    }

    result

}


#[no_mangle]
pub extern fn predict_linear_model_classification(model: *mut Vec<f64>,
                                                  inputs: *mut f64, inputs_size: usize) -> f64 {
    let boxed_model;
    let inputs_slice;

    unsafe {
        //Récupération des contenus des pointeurs
        boxed_model = model.as_ref().unwrap();
        inputs_slice = from_raw_parts(inputs, inputs_size);
    }

    let mut sum = 0.0;

    //prédictions
        for i  in 0..inputs_slice.len() {
            sum += inputs_slice[i] * boxed_model[i];
        }

        let biais = 1.0;

        sum += biais * boxed_model[inputs_slice.len()];

        sign(sum)


}


#[no_mangle]
pub extern fn predict_linear_model_multiclass_classification(model: *mut Vec<f64>,
                                                             inputs: *mut f64, inputs_size: usize, class_count: usize) -> *mut f64 {
    let boxed_model;
    let inputs_slice;

    unsafe {
        boxed_model = model.as_ref().unwrap();
        inputs_slice = from_raw_parts(inputs, inputs_size);
    }

    let mut sum = 0.0;

    // TODO
    for elt in inputs_slice {
        sum += elt;
    }

    let mut result = Vec::with_capacity(class_count);
    result.push(42.0);
    result.push(51.0);
    result.push(69.0);

    let boxed_result = result.into_boxed_slice();
    let boxed_result_leaked = Box::leak(boxed_result);
    return boxed_result_leaked.as_mut_ptr();
}


#[no_mangle]
pub extern fn train_mlp_model_class(model: *mut Vec<f64>, number_hidden_layer: usize, neurones_count: *mut i32,  inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                output: *mut f64, output_size: usize, output_sample_size: usize, learning_rate: f64)
{
    let mut boxed_model;
    let mut input_slice;
    let mut output_slice;
    let mut neurones_count_slice;
    let mut neurones_values:Vec<Vec<f64>> = Vec::with_capacity(number_hidden_layer);


    unsafe {
        boxed_model = &mut *model;
        input_slice = from_raw_parts(inputs, input_size);
        output_slice = from_raw_parts(output, output_size);
        neurones_count_slice = from_raw_parts(neurones_count, number_hidden_layer);
    }


    for i in 0..neurones_count_slice.len(){
        let mut tmp= Vec::with_capacity(i);
        for k in 0..i{
            tmp.push(0.0);
        }
        neurones_values.push(tmp);
    }

    let mut tmp:Vec<f64> = Vec::with_capacity(input_sample_size);
    for i in 0..input_sample_size{
        tmp.push(0.0);
    }
    let mut errors:Vec<Vec<f64>> = Vec::with_capacity(number_hidden_layer + 1);
    errors.push(tmp);
    for i in 0..neurones_values.len()
    {
        errors.push(neurones_values[i].clone());
    }

    let dataset_size = output_size/output_sample_size;


    let mut result=0.0;
    let biais = 1.0;

    for i in 0..dataset_size{



        for j in (i * output_sample_size).. (i * output_sample_size + output_sample_size)
        {

            //TODO: Optimiser la prédictions ici
            //Prédictions


            for hidden_layer in 0..number_hidden_layer
            {
                for computed_neurones in 0..neurones_count_slice[hidden_layer]
                {
                    result=0.0;
                    if number_hidden_layer == 0
                    {
                        for input_slice_index in 0..input_sample_size
                        {
                            result += boxed_model[0*number_hidden_layer+input_slice_index] * input_slice[i*input_sample_size+input_slice_index];
                        }
                        //Ajout biais
                        result += boxed_model[0*number_hidden_layer+input_sample_size] * biais;
                    }

                    else
                    {   if hidden_layer == number_hidden_layer - 1
                        {
                            for neurones in 0..neurones_count_slice[number_hidden_layer] as usize
                            {
                                result += boxed_model[number_hidden_layer * number_hidden_layer+neurones] * neurones_values[number_hidden_layer][neurones];
                            }
                            //Ajout biais
                            result += boxed_model[number_hidden_layer * number_hidden_layer + neurones_count_slice[number_hidden_layer] as usize];
                        }
                        else
                        {
                            for neurones in 0..neurones_count_slice[hidden_layer-1] as usize
                            {
                                result += boxed_model[hidden_layer-1 * number_hidden_layer + neurones] * neurones_values[hidden_layer-1][neurones];
                            }
                            //Ajout biais
                            result += boxed_model[hidden_layer-1 * number_hidden_layer + neurones_count_slice[hidden_layer-1] as usize];
                        }


                    }

                    neurones_values[hidden_layer][computed_neurones as usize] = sign(neurones_values[hidden_layer][computed_neurones as usize]);

                }
            }


            let mut output_error = (1.0 - (result * result)) * (result - output_slice[j]);

            let mut sum = 0.0;
            for computed_neurones in 0..neurones_count_slice[(number_hidden_layer-1)]
            {
                sum += boxed_model[(number_hidden_layer-1) * number_hidden_layer + computed_neurones as usize] * output_error;

            }
            for computed_neurones in 0..neurones_count_slice[(number_hidden_layer-1)]
            {
                errors[(number_hidden_layer-1)][computed_neurones as usize] = (1.0 - power(neurones_values[(number_hidden_layer-1)][computed_neurones as usize], 2)) * sum;
            }

            for hidden_layer in (number_hidden_layer..0).rev()
            {
                sum = 0.0;
                if hidden_layer == 0
                {
                    for computed_neurones in 0..neurones_count_slice[hidden_layer]
                    {
                        sum += boxed_model[hidden_layer * number_hidden_layer + computed_neurones as usize] * errors[hidden_layer][computed_neurones as usize];

                    }

                    for computed_neurones in 0..input_sample_size
                    {
                        errors[0][computed_neurones] = (1.0 - power(neurones_values[(hidden_layer-1)][computed_neurones as usize], 2)) * sum;
                    }
                }
                else
                {

                    for computed_neurones in 0..neurones_count_slice[hidden_layer]
                    {
                        sum += boxed_model[hidden_layer * number_hidden_layer + computed_neurones as usize] * errors[hidden_layer][computed_neurones as usize];

                    }

                    for computed_neurones in 0..neurones_count_slice[hidden_layer-1]
                    {
                        errors[(hidden_layer-1)][computed_neurones as usize] = (1.0 - power(neurones_values[(hidden_layer-1)][computed_neurones as usize], 2)) * sum;
                    }
                }
            }

            for input in 0..input_sample_size
            {
                boxed_model[0*number_hidden_layer+i] = boxed_model[0*number_hidden_layer+i] - learning_rate * input_slice[i * input_sample_size + input] * errors[0][input];
            }

            for HL in 0..number_hidden_layer
            {
                for neurones in 0..neurones_count_slice[i]
                {
                    boxed_model[HL*number_hidden_layer+neurones as usize] = boxed_model[HL*number_hidden_layer+i] - learning_rate * neurones_values[HL][neurones as usize] * errors[HL][neurones as usize];
                }
            }

        }

    }
}


#[no_mangle]
pub extern fn train_model_class(model: *mut Vec<f64>, inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                output: *mut f64, output_size: usize, output_sample_size: usize, learning_rate: f64)
{
    let mut boxed_model;
    let mut input_slice;
    let mut output_slice;
    unsafe {
        boxed_model = &mut *model;
        input_slice = from_raw_parts(inputs, input_size);
        output_slice = from_raw_parts(output, output_size);
    }

    let dataset_size = output_size/output_sample_size;


    let mut result=0.0;
    let biais = 1.0;

    for i in 0..dataset_size{


        for j in (i * output_sample_size).. (i * output_sample_size + output_sample_size){

            //TODO: Optimiser la prédictions ici
            //Prédictions
            result=0.0;
            for k in (i * input_sample_size).. (i * input_sample_size + input_sample_size){

                result += input_slice[k]*boxed_model[k%input_sample_size];

            }
            result += biais*boxed_model[input_sample_size];

            result = sign(result);


            //Mise a jour des poids
            // W[i] = W[i] + r * error * Input[i]
            if result != output_slice[j]
            {
                let mut last_index = 0;
                for l in (i * input_sample_size).. (i * input_sample_size + input_sample_size)
                {
                    boxed_model[l%input_sample_size] = boxed_model[l%input_sample_size] + learning_rate * (output_slice[j] - result) * input_slice[l];
                    last_index = l;
                }
                boxed_model[(last_index%input_sample_size) + 1] = boxed_model[(last_index%input_sample_size) + 1] + learning_rate * (output_slice[j] - result);

            }

        }
    }

}

#[no_mangle]
pub extern fn delete_native_array(arr: *mut f64) {
    unsafe {
        Box::from_raw(arr);
    }
}


#[no_mangle]
pub extern fn delete_linear_model(model: *mut Vec<f64>) {
    unsafe {
        let boxed_model = Box::from_raw(model);
    };
}


#[no_mangle]
pub extern fn my_add(a: f64, b: f64) -> f64 {
    a + b
}