module MNIST.Prelude
  ( module X
  , UVector
  , GVector
  , identity
  ) where

import Prelude                       as X hiding (id)
import Control.Arrow                 as X
import Control.Exception.Safe        as X
import Control.Monad                 as X (zipWithM, when, forM_)
import Control.Monad.IO.Class        as X (liftIO)
import Control.Monad.Trans           as X (lift)
import Control.Monad.Trans.Maybe     as X (MaybeT(..))
import Control.Monad.Trans.Reader    as X hiding (liftCallCC, liftCatch)
import Control.Monad.Trans.State     as X
import Control.DeepSeq               as X
import Criterion.Main                as X
import Criterion.Types               as X
import Data.Bitraversable            as X
import Data.Foldable                 as X
import Data.IDX                      as X
import Data.Int                      as X (Int32, Int64)
import Data.List                     as X (genericLength)
import Data.List.Split               as X
import Data.Maybe                    as X
import Data.Monoid                   as X
import Text.Printf                   as X
import Data.Time.Clock               as X
import Data.Time.Format              as X
import Data.Time.LocalTime           as X
import Data.Traversable              as X
import Data.Tuple                    as X
import Data.Type.Combinator          as X
import Data.Type.Index               as X
import Data.Type.Product             as X hiding (toList)
import Data.Vector                   as X (Vector)
import GHC.Generics                  as X (Generic)
import GHC.TypeLits                  as X
import GHC.IO                        as X (evaluate)
import Numeric.Backprop              as X (Op, op2')
import System.Directory              as X
import TensorFlow.Core               as X (Tensor, Shape, Build, TensorData, Session)
import Numeric.LinearAlgebra.Static  as X (R, L)

import Data.Vector.Unboxed as UV (Vector)
import Data.Vector.Generic as G (Vector)
import qualified Prelude   as P

type UVector = UV.Vector
type GVector = G.Vector

identity :: a -> a
identity = P.id


