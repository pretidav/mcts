import tensorflow as tf 

def relu_bn(inputs):
    relu1 = tf.keras.activations.relu(inputs)
    bn = tf.keras.layers.BatchNormalization()(relu1)
    return bn

def residual_block(x, filters, kernel_size=3):
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides= (1),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    y = tf.keras.layers.BatchNormalization()(y)
    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.activations.relu(out)

    return out

def value_head(input):
    conv1 = tf.keras.layers.Conv2D(kernel_size=1,
                strides=1,
                filters=1,
                padding="same")(input)

    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    bn1_relu = tf.keras.activations.relu(bn1)

    flat = tf.keras.layers.Flatten()(bn1_relu)

    dense1 = tf.keras.layers.Dense(256)(flat)
    dn_relu = tf.keras.activations.relu(dense1)

    dense2 = tf.keras.layers.Dense(256)(dn_relu)

    return dense2

def policy_head(input):
    conv1 = tf.keras.layers.Conv2D(kernel_size=2,
                strides=1,
                filters=1,
                padding="same")(input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    bn1_relu = tf.keras.activations.relu(bn1)
    flat = tf.keras.layers.Flatten()(bn1_relu)
    return flat

class Connect4NNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # Inputs
        self.input_boards = tf.keras.layers.Input(shape=(self.board_x, self.board_y))
        inputs = tf.keras.layers.Reshape((self.board_x, self.board_y, 1))(self.input_boards)


        bn1 = tf.keras.layers.BatchNormalization()(inputs)
        conv1 = tf.keras.layers.Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)


        for i in range(self.args.num_residual_layers):
                t = residual_block(t, filters=self.args.num_channels)

        self.pi = tf.keras.layers.Dense(self.action_size, activation='softmax', name='pi')(policy_head(t))
        self.v = tf.keras.layers.Dense(1, activation='tanh', name='v')(value_head(t))
        self.calculate_loss()
        self.model = tf.keras.Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=[self.loss_pi ,self.loss_v], optimizer=tf.keras.optimizers.Adam(args.lr))

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)