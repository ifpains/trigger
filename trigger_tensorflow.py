import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.ops import variables


class StandardDeviationPooling(tf.keras.layers.Layer): 
    """
    Classe que implementa a camada de extração dos std's das 
    subimagens
    """
    def __init__(self, pool_size=1,
          strides=1,
          rates=[1,1,1,1],
          padding='VALID',
          data_format=None,
          **kwargs):
        
        super(StandardDeviationPooling, self).__init__(**kwargs)
        self.pool_size=pool_size
        self.strides=[1, strides, strides, 1]
        self.rates=rates
        self.padding=padding
        self.data_format=data_format

    def call(self, input_data): 
    ###Realiza a quebra da imagem em subimagens, calculando o desvio padrão de cada uma
    ###gerando assim um vetor de parâmetros de N features contendo o desvio padrao de cada bloco
    ###
        patches = tf.image.extract_patches(images=input_data,
                                    sizes=[1, self.pool_size, self.pool_size, 1],
                                    strides= self.strides, 
                                    rates=self.rates,
                                    padding=self.padding)

        patch_shape = patches.shape
        patches = tf.reshape(patches, (patch_shape[0], patch_shape[1]*patch_shape[2], self.pool_size**2))
        return tf.math.reduce_std(tf.cast(patches, tf.dtypes.float32), axis=-1)



class CustomThresholdLayer(tf.keras.layers.Layer):
    """
    Classe que implementa a aplicação da Layer de threshold.

    A partir do array passado na variavel threshold_array,
    aplica o threshold no vetor de parametros definido na
    variavel inputs

    """
    def __init__(self, thresholds_array):
        self.kernel = thresholds_array
        super(CustomThresholdLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        nsamples = inputs.shape[0]
        expanded_threshold = tf.repeat(
            tf.reshape(self.kernel, (1, inputs.shape[1])), nsamples, axis=0
        )
        upper_threshold = tf.math.greater_equal(
          inputs, expanded_threshold
        )
        return tf.cast(tf.reduce_any(upper_threshold, axis=-1), tf.dtypes.int32)

    def compute_output_shape(self, input_shape):
        return (1, )



class TriggerModel(tf.keras.Model):
    """Modelo customizado para o sistema de trigger"""

    def __init__(self, pool_size, strides, niqr):
        """Construtor da classe, instancia a layer que faz o calculo dos desvios padrões
        a partir das variaveis que definem o tamanho da janela e tambem o numero de desvios
        padroes utilizados no threshold.
        """
        super(TriggerModel, self).__init__()

        self.std_layer = StandardDeviationPooling(pool_size=pool_size, strides=strides)
        self.niqr = niqr

    def call(self,input_tensor):

        """Metodo call, utilizado para fazer as predicoes do modelo. Aplica sobre o tensor
        de entrada as operacoes de calculo de desvio padrao dos blocos e aplicacao do threshold"""
        x = self.std_layer(input_tensor)
        y = self.threshold_layer(x)

        return y

    def fit(self, X, y=None):
        """Metodo fit, utilizado para realizar o ajuste dos parametros do threshold. Primeiro
        os dados de entrada são utilizados para calculo dos desvios padroes das imagens fornecidas
        na entrada. Após isso, a média e desvio padrao desses desvios são calculados e definidos
        na criacao da Layer de threshold. O valor utilizado é média + n*std para calculo dos thresholds."""
        x = self.std_layer(X)
        x_down = tfp.stats.percentile(
                              x,
                              q = 25,
                              axis=0,
                              interpolation=None,
                              keepdims=False,
                              validate_args=False,
                              preserve_gradients=True,
                              name=None
                          )
        x_up = tfp.stats.percentile(
                                    x,
                                    q = 75,
                                    axis=0,
                                    interpolation=None,
                                    keepdims=False,
                                    validate_args=False,
                                    preserve_gradients=True,
                                    name=None
                                )

        iqr = x_up - x_down

        self.threshold_layer = CustomThresholdLayer(thresholds_array=x_up + self.niqr*iqr)