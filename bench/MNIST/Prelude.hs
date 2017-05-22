module MNIST.Prelude
  ( module X
  , UVector
  , identity
  ) where

import Prelude                 as X hiding (id)
import Control.Monad           as X (zipWithM, when, forM_)
import Control.Monad.IO.Class  as X (liftIO)
import Data.Int                as X (Int32, Int64)
import Data.List               as X (genericLength)
import TensorFlow.Core         as X (Tensor, Shape, Build, TensorData, Session)
import Control.Exception.Safe  as X
import Data.IDX                as X
import Data.Monoid             as X
import Data.Vector             as X (Vector)
import Data.List.Split            as X
import Control.Arrow as X
import Control.Monad.Trans.Maybe  as X

import qualified Prelude as P

import Data.Vector.Unboxed     as UV (Vector)

type UVector = UV.Vector
identity = P.id


