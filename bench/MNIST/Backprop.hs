{-# OPTIONS_GHC -fno-warn-orphans #-}
module MNIST.Backprop where

import MNIST.Prelude
import qualified Data.Vector.Generic                 as VG
import qualified Data.Vector.Unboxed                 as VU
import qualified Generics.SOP                        as SOP
import qualified Numeric.LinearAlgebra               as HM
import qualified System.Random.MWC                   as MWC
import Numeric.LinearAlgebra.Static
import Numeric.Backprop

data Layer i o = Layer
  { _lWeights :: !(L o i)
  , _lBiases  :: !(R o)
  } deriving (Show, Generic)

instance SOP.Generic (Layer i o)
instance NFData (Layer i o)

data Network i h1 h2 o = Net
  { _nLayer1 :: !(Layer i  h1)
  , _nLayer2 :: !(Layer h1 h2)
  , _nLayer3 :: !(Layer h2 o)
  } deriving (Show, Generic)

instance SOP.Generic (Network i h1 h2 o)
instance NFData (Network i h1 h2 o)


matVec :: forall m n . (KnownNat m, KnownNat n) => Op '[ L m n, R n ] (R m)
matVec = op2' $ \m v -> (forward m v, backward m v)
  where
    forward :: L m n -> R n -> R m
    forward m v = m #> v

    backward :: L m n -> R n -> Maybe (R m) -> (L m n, R n)
    backward m v (fromMaybe 1 -> g) = (g `outer` v, tr m #> g)


dot :: forall n . KnownNat n => Op '[ R n, R n ] Double
dot = op2' $ \x y -> (forward x y, backward x y)
  where
    forward :: R n -> R n -> Double
    forward x y = x <.> y

    backward :: R n -> R n -> Maybe Double -> (R n, R n)
    backward x y = \case
      Nothing -> (y, x)
      Just g  -> (konst g * y, x * konst g)


scale :: forall n . KnownNat n => Op '[ Double, R n ] (R n)
scale = op2' $ \a x -> (forward a x, backward a x)
  where
    forward :: Double -> R n -> R n
    forward a x = konst a * x

    backward :: Double -> R n -> Maybe (R n) -> (Double, R n)
    backward a x = \case
      Nothing -> (HM.sumElements (extract x      ), konst a    )
      Just g  -> (HM.sumElements (extract (x * g)), konst a * g)


vsum :: forall n . KnownNat n => Op '[ R n ] Double
vsum = op1' $ \x -> (forward x, backward)
  where
    forward :: R n -> Double
    forward = HM.sumElements . extract

    backward :: Maybe Double -> R n
    backward = maybe 1 konst


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

