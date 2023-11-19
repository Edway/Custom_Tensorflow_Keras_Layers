import tensorflow as tf


class SlidingWindowMemory( tf.keras.layers.Layer ):
	
	def __init__( self, length: int, **kwargs ):
		super().__init__( **kwargs )
		
		self.memory = None
		self.size = length
		self.weight = tf.Variable( tf.ones( length ) )
	
	def build( self, input_shape ):
		memory_shape = [ a for a in input_shape ]
		memory_shape.insert( 1, self.size )
		self.memory = tf.Variable( tf.zeros( memory_shape[ 1:: ] ) )  # don't need batch_size for memory
		return super().build( input_shape )
	
	def call( self, inputs, training=True, *args, **kwargs ):
		
		if tf.shape( inputs )[ 0 ] is None:
			return self.memory
		else:
			# shift into memory and output what is memory each step of input length
			outs = tf.TensorArray( tf.float32, size=tf.shape( inputs )[ 0 ], dynamic_size=True )
			
			for d in range( tf.shape( inputs )[ 0 ] ):
				self.memory = tf.concat( [ tf.expand_dims( inputs[ d ], 0 ), self.memory[ 1:: ] ], 0 )
				outs.write( d, self.memory )
			
			return outs.stack()
